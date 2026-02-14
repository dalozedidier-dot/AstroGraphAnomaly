#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — A→H Visualization Suite
------------------------------------------
Implements the A→H gallery:
A) Hidden Constellations-style sky map (density cloud + glow + organic curves)
B) Celestial sphere 3D (Plotly)
C) Network explorer (PyVis)
D) Explainability heatmaps (LIME jsonl if available, else robust z-scores)
E) Simple dashboard (HTML index linking to all outputs)
F) Proper motion trails (GIF)
G) "Biocubes"-style feature summary (Plotly 3D)
H) UMAP cosmic cloud (PNG + Plotly HTML)

Workflow-first:
- Standalone script (no package import).
- Reads `scored.csv` + optional graph/explanations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional deps
try:
    from scipy.ndimage import gaussian_filter  # type: ignore
    _HAS_SCIPY = True
except Exception:
    gaussian_filter = None
    _HAS_SCIPY = False

try:
    import plotly.graph_objects as go  # type: ignore
    import plotly.io as pio  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    go = None
    pio = None
    _HAS_PLOTLY = False
try:
    from pyvis.network import Network  # type: ignore
    _HAS_PYVIS = True
except Exception:
    Network = None
    _HAS_PYVIS = False

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    umap = None
    _HAS_UMAP = False

try:
    import imageio.v2 as imageio  # type: ignore
    _HAS_IMAGEIO = True
except Exception:
    imageio = None
    _HAS_IMAGEIO = False



def _require_viz_deps(allow_missing: bool) -> None:
    """Fail fast if interactive deps are missing, unless allow_missing is set.

    This avoids the confusing situation where the suite 'succeeds' but produces tiny placeholder HTML.
    """
    missing = []
    if not _HAS_PLOTLY:
        missing.append("plotly")
    if not _HAS_PYVIS:
        missing.append("pyvis")
    if not _HAS_IMAGEIO:
        missing.append("imageio")
    if missing and not allow_missing:
        raise SystemExit(
            "Missing visualization dependencies: "
            + ", ".join(missing)
            + ". Install them with: pip install -r requirements_viz.txt"
        )


def robust_unit_interval(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        lo = np.nanmin(x)
        hi = np.nanmax(x)
        if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
            return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) + 1e-12
    z = (x - med) / (1.4826 * mad)
    z = np.where(np.isfinite(z), z, 0.0)
    return z


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype(str)

    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0.0
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)

    if "anomaly_label" not in df.columns:
        thr = np.nanpercentile(df["anomaly_score"].to_numpy(float), 95)
        df["anomaly_label"] = np.where(df["anomaly_score"].to_numpy(float) >= thr, -1, 1)

    s = df["anomaly_score"].to_numpy(float)
    y = df["anomaly_label"].to_numpy(int)
    if np.mean(s[y == -1]) < np.mean(s[y == 1]):
        df["anomaly_score_hi"] = -df["anomaly_score"]
    else:
        df["anomaly_score_hi"] = df["anomaly_score"]

    df["anomaly_score_norm"] = robust_unit_interval(df["anomaly_score_hi"].to_numpy(float))

    if "distance" not in df.columns and "parallax" in df.columns:
        par = pd.to_numeric(df["parallax"], errors="coerce")
        df["distance"] = (1000.0 / par).replace([np.inf, -np.inf], np.nan)

    return df



# -----------------------------
# Multi-color "incoherence" model
# -----------------------------

_DEFAULT_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
]


def _parse_weights(spec: str) -> Dict[str, float]:
    """
    Parse weights like: "graph=2.0,lof=1,ocsvm=1"
    Keys are names WITHOUT the phi_ prefix.
    """
    out: Dict[str, float] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            continue
    return out


def _resolve_ra_dec_cols(df: pd.DataFrame) -> Optional[Tuple[str, str]]:
    """
    Accept common variants: ra/dec, ra_deg/dec_deg, ra_mean/dec_mean.
    """
    candidates = [
        ("ra", "dec"),
        ("ra_deg", "dec_deg"),
        ("ra_mean", "dec_mean"),
        ("ra_icrs", "dec_icrs"),
    ]
    for a, b in candidates:
        if a in df.columns and b in df.columns:
            return a, b
    return None


def _apply_viz_colors(
    df: pd.DataFrame,
    color_mode: str = "auto",
    phi_prefix: str = "phi_",
    phi_weights_spec: str = "",
    rgb_phis_spec: str = "",
) -> pd.DataFrame:
    """
    Adds:
      - viz_color_value (float 0..1)
      - viz_color (optional rgb/hex string)
      - viz_size (float)
      - viz_category (dominant incoherence label)
    Behavior:
      - auto: if >=2 phi_* cols exist -> dominant_phi else score
      - score: continuous anomaly_score_norm
      - dominant_phi: discrete palette by dominant weighted phi
      - rgb_phi: rgb blend of 3 selected phi channels
    """
    d = df.copy()
    d["viz_category"] = ""
    d["viz_color"] = np.nan
    d["viz_color_value"] = d.get("anomaly_score_norm", 0.0)
    d["viz_size"] = 3.0 + 10.0 * pd.to_numeric(d["viz_color_value"], errors="coerce").fillna(0.0).to_numpy(float)

    phi_cols = [c for c in d.columns if c.startswith(phi_prefix)]
    mode = color_mode.strip().lower()
    if mode == "auto":
        mode = "dominant_phi" if len(phi_cols) >= 2 else "score"

    if mode == "score":
        d["viz_color_value"] = pd.to_numeric(d.get("anomaly_score_norm", 0.0), errors="coerce").fillna(0.0)
        d["viz_size"] = 3.0 + 10.0 * d["viz_color_value"].to_numpy(float)
        return d

    if len(phi_cols) < 1:
        # fallback
        d["viz_color_value"] = pd.to_numeric(d.get("anomaly_score_norm", 0.0), errors="coerce").fillna(0.0)
        d["viz_size"] = 3.0 + 10.0 * d["viz_color_value"].to_numpy(float)
        return d

    weights = _parse_weights(phi_weights_spec)
    # Normalize each phi column robustly into [0,1]
    phi_norm = []
    w_list = []
    labels = []
    for c in phi_cols:
        lab = c[len(phi_prefix):]
        labels.append(lab)
        w = float(weights.get(lab, 1.0))
        w_list.append(w)
        arr = pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(float)
        phi_norm.append(robust_unit_interval(arr))
    phi_norm = np.vstack(phi_norm)  # (k, n)
    w_arr = np.asarray(w_list, dtype=float)[:, None]
    contrib = phi_norm * w_arr
    denom = float(np.sum(w_arr)) if float(np.sum(w_arr)) > 0 else 1.0
    composite = np.sum(contrib, axis=0) / denom
    composite = np.clip(composite, 0.0, 1.0)

    d["viz_color_value"] = composite
    d["viz_size"] = 3.0 + 10.0 * composite

    if mode == "dominant_phi":
        idx = np.argmax(contrib, axis=0)
        dom = [labels[int(i)] for i in idx]
        d["viz_category"] = dom

        uniq = sorted(set(dom))
        cmap = {k: _DEFAULT_PALETTE[i % len(_DEFAULT_PALETTE)] for i, k in enumerate(uniq)}
        d["viz_color"] = [cmap[k] for k in dom]
        return d

    if mode == "rgb_phi":
        # Pick 3 channels
        rgb_names = [s.strip() for s in rgb_phis_spec.split(",") if s.strip()]
        if len(rgb_names) < 3:
            rgb_names = labels[:3]
        rgb_names = (rgb_names + labels * 3)[:3]

        def get_phi(name: str) -> np.ndarray:
            if name in labels:
                j = labels.index(name)
                return phi_norm[j]
            # fallback to composite if missing
            return composite

        r = get_phi(rgb_names[0])
        g = get_phi(rgb_names[1])
        b = get_phi(rgb_names[2])
        cols = [f"rgb({int(255*rv)},{int(255*gv)},{int(255*bv)})" for rv, gv, bv in zip(r, g, b)]
        d["viz_category"] = "rgb(" + ",".join(rgb_names) + ")"
        d["viz_color"] = cols
        return d

    # unknown -> score
    d["viz_color_value"] = pd.to_numeric(d.get("anomaly_score_norm", 0.0), errors="coerce").fillna(0.0)
    d["viz_size"] = 3.0 + 10.0 * d["viz_color_value"].to_numpy(float)
    return d


def _placeholder_png(path: Path, title: str, msg: str) -> None:
    fig = plt.figure(figsize=(10, 6), dpi=140)
    plt.axis("off")
    plt.text(0.5, 0.65, title, ha="center", va="center", fontsize=16)
    plt.text(0.5, 0.45, msg, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _placeholder_html(path: Path, title: str, msg: str) -> None:
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui;background:#07080c;color:#eaeaea;margin:24px}}pre{{white-space:pre-wrap}}</style>
</head><body><h1>{title}</h1><pre>{msg}</pre></body></html>"""
    path.write_text(html, encoding="utf-8")


def _write_plotly_html(fig, out_html: Path, title: str) -> None:
    """Write a robust offline Plotly HTML.

    Uses include_plotlyjs='directory' so plotly.min.js is written once in the output
    folder (small artifacts) while figure data remain embedded in the HTML.
    """
    if not _HAS_PLOTLY or pio is None:
        _placeholder_html(out_html, title, "Plotly not installed. Install requirements_viz.txt.")
        return

    out_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        # If a 3D scene exists, prefer a turntable dragmode for axis-aligned rotation.
        fig.update_layout(scene=dict(dragmode="turntable"))
    except Exception:
        pass

    pio.write_html(
        fig,
        file=str(out_html),
        include_plotlyjs="directory",
        full_html=True,
        auto_open=False,
    )

def _placeholder_gif(path: Path, title: str, msg: str) -> None:
    if not _HAS_IMAGEIO:
        # fallback
        _placeholder_html(path.with_suffix(".html"), title, msg)
        return
    tmp_png = path.with_suffix(".png")
    _placeholder_png(tmp_png, title, msg)
    img = imageio.imread(tmp_png)
    imageio.mimsave(path, [img], duration=0.5)
    try:
        tmp_png.unlink()
    except Exception:
        pass


def _safe_call(name: str, outputs: List[Tuple[Path, str]], fn, profile: Optional[dict] = None) -> None:
    """
    outputs: list of (path, kind) where kind in {"png","html","gif"}.
    """
    try:
        fn()
        if profile is not None:
            profile.setdefault('steps', {})[name] = {'ok': True, 'error': None, 'outputs': [str(p) for p, _ in outputs]}
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        if profile is not None:
            profile.setdefault('steps', {})[name] = {'ok': False, 'error': msg, 'outputs': [str(p) for p, _ in outputs]}
        for p, kind in outputs:
            try:
                if kind == "png":
                    _placeholder_png(p, name, msg)
                elif kind == "gif":
                    _placeholder_gif(p, name, msg)
                else:
                    _placeholder_html(p, name, msg)
            except Exception:
                pass
        print(f"WARNING: {name} failed -> placeholders written. {msg}")


def pick_graph_path(run_dir: Path, arg_graph: Optional[str]) -> Optional[Path]:
    if arg_graph:
        p = Path(arg_graph)
        return p if p.exists() else None
    for name in ["graph_union.graphml", "graph_full.graphml", "graph_topk.graphml"]:
        p = run_dir / name
        if p.exists():
            return p
    return None


def load_graph(graph_path: Path):
    import networkx as nx
    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})
    return G


def community_labels(G) -> Dict[str, int]:
    comm_id: Dict[str, int] = {}
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G)
    for i, cset in enumerate(comms):
        for n in cset:
            comm_id[str(n)] = int(i)
    return comm_id


def sample_subgraph(G, df: pd.DataFrame, max_nodes: int = 1200) -> Tuple[List[str], List[Tuple[str, str]]]:
    rng = np.random.default_rng(42)
    nodes = list(G.nodes())
    if len(nodes) <= max_nodes:
        edges = [(str(u), str(v)) for u, v in G.edges()]
        return [str(n) for n in nodes], edges

    d = df.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d[d["source_id"].isin(nodes)]
    d = d.sort_values("anomaly_score_hi", ascending=False)

    top_keep = d.head(min(200, len(d)))["source_id"].astype(str).tolist()
    comm = {}
    if "community_id" in d.columns:
        comm = dict(zip(d["source_id"].astype(str).tolist(), d["community_id"].to_numpy(int)))

    keep = set(top_keep)
    if comm:
        for cid in sorted(set(comm.values())):
            pool = d[(d["community_id"] == cid) & (d["anomaly_label"].astype(int) != -1)]["source_id"].astype(str).tolist()
            if len(pool) == 0:
                continue
            k = min(25, len(pool))
            chosen = rng.choice(pool, size=k, replace=False).tolist()
            keep.update(chosen)

    if len(keep) < max_nodes:
        pool = [n for n in nodes if n not in keep]
        k = min(max_nodes - len(keep), len(pool))
        if k > 0:
            keep.update(rng.choice(pool, size=k, replace=False).tolist())

    keep_list = sorted(keep)
    keep_set = set(keep_list)
    edge_list = []
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su in keep_set and sv in keep_set:
            edge_list.append((su, sv))
    return keep_list, edge_list


def _blur2d(H: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    if _HAS_SCIPY and gaussian_filter is not None:
        return gaussian_filter(H, sigma=sigma)
    K = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float)
    K /= K.sum()
    X = H.copy()
    for _ in range(3):
        Xp = np.pad(X, 1, mode="edge")
        Y = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Y[i, j] = np.sum(Xp[i:i+3, j:j+3] * K)
        X = Y
    return X


def _glow_scatter(x, y, c, s, ax):
    ax.scatter(x, y, s=s*10, c=c, alpha=0.08, linewidths=0)
    ax.scatter(x, y, s=s*4,  c=c, alpha=0.12, linewidths=0)
    ax.scatter(x, y, s=s,    c=c, alpha=0.65, linewidths=0)


def plot_hidden_constellations(df: pd.DataFrame, G_opt, out_png: Path) -> None:
    if "ra" not in df.columns or "dec" not in df.columns:
        fig = plt.figure(figsize=(12, 7))
        plt.text(0.5, 0.5, "Missing ra/dec", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        return

    ra = pd.to_numeric(df["ra"], errors="coerce").fillna(0.0).to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").fillna(0.0).to_numpy(float)
    score = df["anomaly_score_norm"].to_numpy(float)

    bins = 320
    xedges = np.linspace(np.nanmin(ra), np.nanmax(ra), bins+1)
    yedges = np.linspace(np.nanmin(dec), np.nanmax(dec), bins+1)
    H, _, _ = np.histogram2d(ra, dec, bins=[xedges, yedges])
    H = _blur2d(H.T, sigma=1.6)

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.imshow(H, origin="lower", aspect="auto",
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
              alpha=0.95)

    _glow_scatter(ra, dec, score, s=8, ax=ax)

    if G_opt is not None:
        G = G_opt
        comm = community_labels(G)
        sid = df["source_id"].astype(str) if "source_id" in df.columns else df.index.astype(str)
        df_idx = dict(zip(sid.tolist(), range(len(df))))
        d_sorted = df.sort_values("anomaly_score_hi", ascending=False)
        top_ids = set(d_sorted.head(min(120, len(d_sorted)))["source_id"].astype(str).tolist()) if "source_id" in df.columns else set()

        kept = 0
        for u, v in G.edges():
            su, sv = str(u), str(v)
            if su not in df_idx or sv not in df_idx:
                continue
            if comm.get(su, -1) != comm.get(sv, -2) and not (su in top_ids or sv in top_ids):
                continue
            if kept > 2200:
                break
            iu, iv = df_idx[su], df_idx[sv]
            x1, y1 = ra[iu], dec[iu]
            x2, y2 = ra[iv], dec[iv]
            mx, my = 0.5*(x1+x2), 0.5*(y1+y2)
            dx, dy = (x2-x1), (y2-y1)
            norm = math.hypot(dx, dy) + 1e-9
            px, py = (-dy/norm, dx/norm)
            offset = 0.08 * norm
            cx, cy = mx + px*offset, my + py*offset
            t = np.linspace(0, 1, 20)
            bx = (1-t)**2 * x1 + 2*(1-t)*t * cx + t**2 * x2
            by = (1-t)**2 * y1 + 2*(1-t)*t * cy + t**2 * y2
            w = 0.6 if (su in top_ids or sv in top_ids) else 0.35
            a = 0.22 if (su in top_ids or sv in top_ids) else 0.12
            ax.plot(bx, by, linewidth=w, alpha=a)
            kept += 1

    ax.set_title("Hidden Constellations (Gaia graph reinterpretation)")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.grid(alpha=0.10)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)



def export_celestial_sphere(df: pd.DataFrame, out_html: Path) -> None:
    """
    B) Celestial sphere 3D interactive (Plotly).
    Offline-first: embeds plotly.js directly in the HTML so opening the file locally works.
    Multi-color:
      - If viz_color (string) exists -> use it (dominant_phi or rgb_phi).
      - Else -> continuous anomaly_score_norm as colorscale.
    """
    if not _HAS_PLOTLY:
        _placeholder_html(out_html, "Celestial Sphere 3D", "Plotly not installed. Install requirements_viz.txt.")
        return

    cols = _resolve_ra_dec_cols(df)
    if cols is None:
        _placeholder_html(out_html, "Celestial Sphere 3D", "Missing ra/dec columns in scored.csv (expected ra/dec or ra_deg/dec_deg).")
        return
    ra_col, dec_col = cols

    ra = np.deg2rad(pd.to_numeric(df[ra_col], errors="coerce").fillna(0.0).to_numpy(float))
    dec = np.deg2rad(pd.to_numeric(df[dec_col], errors="coerce").fillna(0.0).to_numpy(float))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    size = df["viz_size"].to_numpy(float) if "viz_size" in df.columns else (3 + 10*df["anomaly_score_norm"].to_numpy(float))
    hover_cols = [c for c in ["source_id","viz_category","anomaly_score_hi","phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe","community_id"] if c in df.columns]
    hover = df[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

    # Prefer multi-color if available
    has_viz_color = "viz_color" in df.columns and df["viz_color"].notna().any()
    if has_viz_color:
        colors = df["viz_color"].fillna("#888888").astype(str).to_list()
        marker = dict(size=size, color=colors, opacity=0.88)
        title = "Celestial Sphere — incoherence colors"
    else:
        val = df["viz_color_value"].to_numpy(float) if "viz_color_value" in df.columns else df["anomaly_score_norm"].to_numpy(float)
        marker = dict(
            size=size,
            color=val,
            colorscale="Viridis",
            opacity=0.88,
            colorbar=dict(title="Anomaly / Incoherence"),
        )
        title = "Celestial Sphere — anomaly score"

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=marker,
        text=hover,
        hoverinfo="text" if hover is not None else "skip"
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    _write_plotly_html(fig, out_html, "Celestial Sphere 3D")


def export_network_explorer(df: pd.DataFrame, G_opt, out_html: Path) -> None:
    if not _HAS_PYVIS:
        out_html.write_text("PyVis not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    if G_opt is None:
        out_html.write_text("No graph provided (graph_full/union.graphml).", encoding="utf-8")
        return

    G = G_opt
    nodes_keep, edges_keep = sample_subgraph(G, df, max_nodes=1200)

    d = df.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d.set_index("source_id", drop=False)

    net = Network(height="820px", width="100%", bgcolor="#05060a", font_color="#e8e8e8", directed=False)
    net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=110, spring_strength=0.08, damping=0.4)

    for sid in nodes_keep:
        row = d.loc[sid] if sid in d.index else None
        sc = float(row["anomaly_score_norm"]) if row is not None and "anomaly_score_norm" in row else 0.2
        size = 10 + 30*sc
        label = sid if sc > 0.92 else ""
        title_parts = []
        if row is not None:
            cols = ["source_id","anomaly_score_hi","phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe","degree","kcore","betweenness"]
            for c in cols:
                if c in row.index:
                    title_parts.append(f"{c}: {row[c]}")
        title = "<br>".join(title_parts) if title_parts else sid

        if row is not None and "viz_color" in row.index and pd.notna(row["viz_color"]):
            color = str(row["viz_color"])
        else:
            r = int(40 + 215*sc)
            g = int(80 + 150*sc)
            b = int(220 - 160*sc)
            color = f"rgb({r},{g},{b})"
        net.add_node(sid, label=label, title=title, value=size, color=color)

    for u, v in edges_keep:
        net.add_edge(u, v, value=1)

    net.save_graph(str(out_html))


def load_lime_matrix(explain_jsonl: Path, top_ids: List[str]) -> Optional[Tuple[List[str], List[str], np.ndarray]]:
    if not explain_jsonl.exists():
        return None
    rows = []
    feats_set = set()
    try:
        with explain_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sid = str(obj.get("source_id", ""))
                if sid not in set(top_ids):
                    continue
                lime = obj.get("lime") or obj.get("lime_weights") or obj.get("explanation")
                if lime is None:
                    continue
                if isinstance(lime, list):
                    pairs = []
                    for it in lime:
                        if isinstance(it, (list, tuple)) and len(it) >= 2:
                            pairs.append((str(it[0]), float(it[1])))
                    if not pairs:
                        continue
                    for k, _ in pairs:
                        feats_set.add(k)
                    rows.append((sid, pairs))
        if not rows or not feats_set:
            return None
        feat_list = sorted(feats_set)
        M = np.zeros((len(rows), len(feat_list)), dtype=float)
        for i, (sid, pairs) in enumerate(rows):
            w = dict(pairs)
            for j, feat in enumerate(feat_list):
                M[i, j] = float(w.get(feat, 0.0))
        sid_list = [r[0] for r in rows]
        return sid_list, feat_list, M
    except Exception:
        return None


def plot_explainability_heatmap(df: pd.DataFrame, explain_jsonl: Optional[Path], out_png: Path, top_n: int = 40) -> None:
    d = df.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d.sort_values("anomaly_score_hi", ascending=False).head(min(top_n, len(d)))
    top_ids = d["source_id"].astype(str).tolist()

    lime = None
    if explain_jsonl is not None:
        lime = load_lime_matrix(explain_jsonl, top_ids)

    if lime is not None:
        sid_list, feat_list, M = lime
        title = "Explainability heatmap (LIME weights)"
        data = M
        ylabels = sid_list
        xlabels = feat_list
    else:
        num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c]) and c not in ("anomaly_label",)]
        preferred = [c for c in ["phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","distance","ruwe","degree","kcore","betweenness"] if c in num_cols]
        cols = preferred if len(preferred) >= 6 else num_cols[:12]
        Z = np.vstack([robust_z(pd.to_numeric(d[c], errors="coerce").fillna(0.0).to_numpy(float)) for c in cols]).T
        title = "Explainability heatmap (fallback: robust z-scores)"
        data = Z
        ylabels = top_ids
        xlabels = cols

    fig = plt.figure(figsize=(max(12, 0.55*len(xlabels)), max(7, 0.26*len(ylabels))))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("features")
    ax.set_ylabel("top anomalies")
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    plt.colorbar(im, ax=ax, shrink=0.7)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def export_proper_motion_trails(df: pd.DataFrame, out_gif: Path, top_k: int = 30, frames: int = 24) -> None:
    """F) Proper motion trails (GIF)."""
    if not _HAS_IMAGEIO or imageio is None:
        _placeholder_png(out_gif.with_suffix('.png'), "Proper Motion Trails", "imageio not installed. Install requirements_viz.txt.")
        return

    # Columns
    ra_col = 'ra' if 'ra' in df.columns else ('ra_deg' if 'ra_deg' in df.columns else None)
    dec_col = 'dec' if 'dec' in df.columns else ('dec_deg' if 'dec_deg' in df.columns else None)
    if ra_col is None or dec_col is None:
        # single-frame gif placeholder
        tmp = out_gif.with_suffix('.png')
        _placeholder_png(tmp, "Proper Motion Trails", "Missing ra/dec columns.")
        img = imageio.imread(tmp)
        imageio.mimsave(out_gif, [img], duration=0.25)
        return

    if not {'pmra','pmdec'}.issubset(df.columns):
        tmp = out_gif.with_suffix('.png')
        _placeholder_png(tmp, "Proper Motion Trails", "Missing pmra/pmdec columns.")
        img = imageio.imread(tmp)
        imageio.mimsave(out_gif, [img], duration=0.25)
        return

    d = df.copy()
    d['anomaly_score_hi'] = pd.to_numeric(d.get('anomaly_score_hi', d.get('anomaly_score', 0.0)), errors='coerce').fillna(0.0)
    d = d.sort_values('anomaly_score_hi', ascending=False).head(max(1, min(top_k, len(d))))

    ra = pd.to_numeric(d[ra_col], errors='coerce').to_numpy(float)
    dec = pd.to_numeric(d[dec_col], errors='coerce').to_numpy(float)
    pmra = pd.to_numeric(d['pmra'], errors='coerce').to_numpy(float)
    pmdec = pd.to_numeric(d['pmdec'], errors='coerce').to_numpy(float)

    # Convert mas/yr -> deg per yr
    mas_to_deg = 1.0 / 3.6e6
    dt_years = 6.0
    dra = pmra * dt_years * mas_to_deg
    ddec = pmdec * dt_years * mas_to_deg

    # Normalize view window
    x0, y0 = ra, dec
    x1, y1 = ra + dra, dec + ddec
    xmin = float(np.nanmin(np.r_[x0, x1]))
    xmax = float(np.nanmax(np.r_[x0, x1]))
    ymin = float(np.nanmin(np.r_[y0, y1]))
    ymax = float(np.nanmax(np.r_[y0, y1]))
    pad_x = (xmax - xmin) * 0.08 + 1e-9
    pad_y = (ymax - ymin) * 0.08 + 1e-9

    frames_list = []
    for k in range(frames):
        t = k / max(1, frames - 1)
        xt = x0 + t * dra
        yt = y0 + t * ddec

        fig = plt.figure(figsize=(8, 6), dpi=140)
        ax = plt.gca()
        ax.set_facecolor('#07080c')
        fig.patch.set_facecolor('#07080c')

        for i in range(len(x0)):
            ax.plot([x0[i], xt[i]], [y0[i], yt[i]], alpha=0.32, linewidth=1.0)
        ax.scatter(xt, yt, s=16, alpha=0.92)

        ax.set_title('Proper Motion Trails', color='white')
        ax.set_xlim(xmin - pad_x, xmax + pad_x)
        ax.set_ylim(ymin - pad_y, ymax + pad_y)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(fig.canvas.buffer_rgba())
        # Matplotlib versions differ: sometimes this is (h, w, 4), sometimes a flat buffer.
        if getattr(buf, "ndim", 1) == 1:
            buf = buf.reshape(h, w, 4)
        img = buf[..., :3].copy()
        frames_list.append(img)
        plt.close(fig)

    out_gif.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_gif, frames_list, duration=0.08)


def export_feature_biocubes(df: pd.DataFrame, out_html: Path) -> None:
    """G) 3D feature biocubes (Plotly)."""
    if not _HAS_PLOTLY or go is None:
        _placeholder_html(out_html, 'Feature BioCubes 3D', 'Plotly not installed. Install requirements_viz.txt.')
        return

    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    prefs = [c for c in ['phot_g_mean_mag','bp_rp','parallax','pmra','pmdec','ruwe','degree','kcore'] if c in num_cols]
    cols = prefs[:3] if len(prefs) >= 3 else num_cols[:3]
    if len(cols) < 3:
        _placeholder_html(out_html, 'Feature BioCubes 3D', 'Not enough numeric columns for a 3D plot.')
        return

    X = df[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    val = pd.to_numeric(df.get('viz_color_value', df.get('anomaly_score_norm', 0.0)), errors='coerce').fillna(0.0).to_numpy(float)

    has_viz_color = 'viz_color' in df.columns and df['viz_color'].notna().any()
    if has_viz_color:
        color = df['viz_color'].fillna('#888888').astype(str).to_list()
        marker = dict(size=3, color=color, opacity=0.85)
    else:
        marker = dict(size=3, color=val, colorscale='Viridis', opacity=0.85, colorbar=dict(title='Anomaly / Incoherence'))

    hover_cols = [c for c in ['source_id','viz_category','anomaly_score_hi'] + cols if c in df.columns]
    hover = df[hover_cols].astype(str).agg('<br>'.join, axis=1) if hover_cols else None

    fig = go.Figure(data=[go.Scatter3d(
        x=X[:,0], y=X[:,1], z=X[:,2],
        mode='markers',
        marker=marker,
        text=hover,
        hoverinfo='text' if hover is not None else 'skip'
    )])
    fig.update_layout(title='Feature BioCubes 3D', scene=dict(
        xaxis_title=cols[0], yaxis_title=cols[1], zaxis_title=cols[2]
    ), margin=dict(l=0, r=0, b=0, t=40))
    _write_plotly_html(fig, out_html, 'Feature BioCubes 3D')


def export_umap(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    """H) UMAP cosmic cloud (PNG + Plotly HTML)."""
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ['phot_g_mean_mag','bp_rp','parallax','pmra','pmdec','distance','ruwe','degree','kcore','betweenness'] if c in num_cols]
    cols = preferred if len(preferred) >= 6 else num_cols[:12]

    if len(cols) < 2:
        _placeholder_png(out_png, 'UMAP cosmic cloud', 'Not enough numeric columns.')
        _placeholder_html(out_html, 'UMAP cosmic cloud', 'Not enough numeric columns.')
        return

    X = df[cols].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    X = np.array(X, dtype=float, copy=True)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    if _HAS_UMAP and umap is not None:
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.18, random_state=42)
        emb = reducer.fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        emb = U[:, :2] * S[:2]

    val = pd.to_numeric(df.get('viz_color_value', df.get('anomaly_score_norm', 0.0)), errors='coerce').fillna(0.0).to_numpy(float)
    has_viz_color = 'viz_color' in df.columns and df['viz_color'].notna().any()

    # PNG
    fig = plt.figure(figsize=(10, 8))
    if has_viz_color:
        colors = df['viz_color'].fillna('#888888').astype(str).to_list()
        plt.scatter(emb[:,0], emb[:,1], s=18, alpha=0.78, c=colors)
        plt.title('UMAP cosmic cloud — incoherence colors')
    else:
        plt.scatter(emb[:,0], emb[:,1], s=18, alpha=0.78, c=val)
        plt.title('UMAP cosmic cloud')
        plt.colorbar(label='Anomaly / Incoherence')
    plt.xlabel('dim-1')
    plt.ylabel('dim-2')
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches='tight')
    plt.close(fig)

    # Plotly HTML
    if not _HAS_PLOTLY or go is None:
        _placeholder_html(out_html, 'UMAP cosmic cloud', 'Plotly not installed. Install requirements_viz.txt.')
        return

    hover_cols = [c for c in ['source_id','viz_category','anomaly_score_hi','phot_g_mean_mag','bp_rp','parallax','pmra','pmdec','ruwe','community_id'] if c in df.columns]
    hover = df[hover_cols].astype(str).agg('<br>'.join, axis=1) if hover_cols else None

    if has_viz_color:
        colors = df['viz_color'].fillna('#888888').astype(str).to_list()
        marker = dict(size=6, color=colors, opacity=0.88)
    else:
        marker = dict(size=6, color=val, colorscale='Viridis', opacity=0.88, colorbar=dict(title='Anomaly / Incoherence'))

    fig2 = go.Figure(data=[go.Scattergl(
        x=emb[:,0], y=emb[:,1],
        mode='markers',
        marker=marker,
        text=hover,
        hoverinfo='text' if hover is not None else 'skip'
    )])
    fig2.update_layout(title='UMAP cosmic cloud (interactive)', margin=dict(l=0, r=0, b=0, t=40))
    _write_plotly_html(fig2, out_html, 'UMAP cosmic cloud')

def plot_feature_interaction_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","distance","ruwe","degree","kcore","betweenness"] if c in num_cols]
    cols = preferred if len(preferred) >= 6 else num_cols[:12]
    if len(cols) < 2:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        return

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    # Some pandas backends can return a read-only view; force a writable copy.
    X = np.array(X, dtype=float, copy=True)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.18, random_state=42)
        emb = reducer.fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        emb = U[:, :2] * S[:2]

    val = df["viz_color_value"].to_numpy(float) if "viz_color_value" in df.columns else df["anomaly_score_norm"].to_numpy(float)
    size = df["viz_size"].to_numpy(float) if "viz_size" in df.columns else (3 + 10*val)

    has_viz_color = "viz_color" in df.columns and df["viz_color"].notna().any()
    if has_viz_color:
        colors = df["viz_color"].fillna("#888888").astype(str).to_list()
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=colors)
        plt.title("Cosmic cloud embedding (UMAP if available) — incoherence colors")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.grid(alpha=0.15)
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
    else:
        fig = plt.figure(figsize=(10, 8))
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=val)
        plt.title("Cosmic cloud embedding (UMAP if available)")
        plt.xlabel("dim-1")
        plt.ylabel("dim-2")
        plt.grid(alpha=0.15)
        plt.colorbar(label="anomaly / incoherence")
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)


def plot_hr_cmd_outliers(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    """
    I) HR/CMD-style outliers (Hertzsprung Russell / Color Magnitude Diagram).

    Preferred axes:
      x = bp_rp (color)
      y = absolute G magnitude (needs parallax and phot_g_mean_mag)

    Robust fallbacks when BP RP is not available:
      x = ruwe (proxy) or parallax (proxy)
      y = phot_g_mean_mag (proxy) or absolute magnitude if possible

    Goal: never return an empty panel for datasets that do not include BP RP bands.
    """
    df = df.copy()

    # X axis
    x = None
    x_label = None
    if "bp_rp" in df.columns:
        x = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy(float)
        x_label = "BP-RP"
    elif {"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(set(df.columns)):
        bp = pd.to_numeric(df["phot_bp_mean_mag"], errors="coerce").to_numpy(float)
        rp = pd.to_numeric(df["phot_rp_mean_mag"], errors="coerce").to_numpy(float)
        x = bp - rp
        x_label = "BP-RP derived"
    elif "ruwe" in df.columns:
        x = pd.to_numeric(df["ruwe"], errors="coerce").to_numpy(float)
        x_label = "RUWE proxy"
    elif "parallax" in df.columns:
        x = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
        x_label = "Parallax mas proxy"
    else:
        _placeholder_png(out_png, "HR/CMD outliers", "Missing x-axis columns.")
        _placeholder_html(out_html, "HR/CMD outliers", "Missing x-axis columns.")
        return

    # Y axis
    y = None
    y_label = None
    if {"phot_g_mean_mag", "parallax"}.issubset(set(df.columns)):
        g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
        plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            y = g + 5.0 * np.log10(plx) - 10.0
        y_label = "M_G abs"
    elif "phot_g_mean_mag" in df.columns:
        y = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
        y_label = "G mag proxy"
    elif "mean_mag_g_fov" in df.columns:
        y = pd.to_numeric(df["mean_mag_g_fov"], errors="coerce").to_numpy(float)
        y_label = "mean_mag_g_fov proxy"
    else:
        _placeholder_png(out_png, "HR/CMD outliers", "Missing y-axis columns.")
        _placeholder_html(out_html, "HR/CMD outliers", "Missing y-axis columns.")
        return

    m = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(m)) < 10:
        _placeholder_png(out_png, "HR/CMD outliers", "Not enough finite points.")
        _placeholder_html(out_html, "HR/CMD outliers", "Not enough finite points.")
        return

    xs = x[m]
    ys = y[m]

    score = pd.to_numeric(df.get("anomaly_score_norm", df.get("anomaly_score", 0.0)), errors="coerce").fillna(0.0).to_numpy(float)
    score = score[m]
    top_n = min(120, max(20, int(0.10 * len(xs))))
    idx = np.argsort(score)[::-1]
    top = idx[:top_n]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8.8, 6.6), dpi=170)
    ax = plt.gca()
    ax.set_facecolor("#07080c")
    fig.patch.set_facecolor("#07080c")
    ax.scatter(xs, ys, s=10, alpha=0.55)
    ax.scatter(xs[top], ys[top], s=26, alpha=0.95)
    ax.set_title("HR/CMD outliers", color="white")
    ax.set_xlabel(x_label, color="white")
    ax.set_ylabel(y_label, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#333")
    try:
        ax.invert_yaxis()
    except Exception:
        pass
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not _HAS_PLOTLY:
        _placeholder_html(out_html, "HR/CMD outliers", "Plotly not installed. Install requirements_viz.txt.")
        return

    try:
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=xs, y=ys, mode="markers", marker=dict(size=5, opacity=0.55), name="all"))
        fig3.add_trace(go.Scattergl(x=xs[top], y=ys[top], mode="markers", marker=dict(size=7, opacity=0.95), name="top anomalies"))
        fig3.update_layout(title="HR/CMD outliers", xaxis_title=x_label, yaxis_title=y_label, template="plotly_dark")
        fig3.update_yaxes(autorange="reversed")
        _write_plotly_html(fig3, out_html, "HR/CMD outliers")
    except Exception as e:
        _placeholder_html(out_html, "HR/CMD outliers", f"Plotly export failed: {e}")

def compute_viz_profile(df: pd.DataFrame, G: Optional[object], graph_path: Optional[Path], args: argparse.Namespace) -> Dict[str, object]:
    """Capture what data is available and which visualization backends are active.

    Purpose: keep the suite utility-first. The same detector output should be visualized
    with the most meaningful representation given the available physical/astronomical variables.
    """
    cols = set(df.columns)
    has_bp_rp = ("bp_rp" in cols) or ({"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(cols))
    has_pm = {"pmra", "pmdec"}.issubset(cols)
    has_parallax = "parallax" in cols
    has_ruwe = "ruwe" in cols

    if "bp_rp" in cols:
        hrcmd_x = "bp_rp"
    elif {"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(cols):
        hrcmd_x = "bp_rp_derived"
    elif has_ruwe:
        hrcmd_x = "ruwe_proxy"
    elif has_parallax:
        hrcmd_x = "parallax_proxy"
    else:
        hrcmd_x = "missing"

    if {"phot_g_mean_mag", "parallax"}.issubset(cols):
        hrcmd_y = "abs_mag_g"
    elif "phot_g_mean_mag" in cols:
        hrcmd_y = "g_mag_proxy"
    elif "mean_mag_g_fov" in cols:
        hrcmd_y = "mean_mag_g_fov_proxy"
    else:
        hrcmd_y = "missing"

    profile: Dict[str, object] = {
        "run_dir": str(Path(args.run_dir)),
        "scored": str(Path(args.scored)),
        "graph": str(graph_path) if graph_path is not None else None,
        "n_rows": int(len(df)),
        "has_plotly": bool(_HAS_PLOTLY),
        "has_pyvis": bool(_HAS_PYVIS),
        "has_umap": bool(_HAS_UMAP),
        "umap_backend": "umap" if (_HAS_UMAP and umap is not None) else "svd",
        "has_bp_rp": bool(has_bp_rp),
        "has_pm": bool(has_pm),
        "has_parallax": bool(has_parallax),
        "has_ruwe": bool(has_ruwe),
        "hrcmd_x": hrcmd_x,
        "hrcmd_y": hrcmd_y,
        "graph_nodes": int(G.number_of_nodes()) if hasattr(G, "number_of_nodes") else 0,
        "graph_edges": int(G.number_of_edges()) if hasattr(G, "number_of_edges") else 0,
        "color_mode": str(getattr(args, "color_mode", "auto")),
        "phi_prefix": str(getattr(args, "phi_prefix", "phi_")),
    }
    return profile


def export_diagnostics(df: pd.DataFrame, out_dir: Path, profile: Dict[str, object]) -> None:
    """Utility-first diagnostics: correlations + top outliers + targeted plots.

    Goal: make sure the detection signal (score) is traceable back to measurable variables,
    not just a black-box scalar.
    """
    diag = out_dir / "diagnostics"
    diag.mkdir(parents=True, exist_ok=True)

    (diag / "viz_profile.json").write_text(json.dumps(profile, indent=2, sort_keys=True), encoding="utf-8")

    score_col = "anomaly_score_norm" if "anomaly_score_norm" in df.columns else ("anomaly_score" if "anomaly_score" in df.columns else None)
    if score_col is None:
        (diag / "note.txt").write_text("No anomaly_score column found; diagnostics limited.", encoding="utf-8")
        for k in (1, 2, 3):
            _placeholder_png(diag / f"scatter_score_vs_feature_{k}.png", "Diagnostics", "No anomaly_score column.")
        return

    s = pd.to_numeric(df[score_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    num_cols = [c for c in df.columns if c not in {score_col, "viz_color_value"} and pd.api.types.is_numeric_dtype(df[c])]
    rows: List[Tuple[str, float, float, int]] = []
    for c in num_cols:
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        m = x.notna() & s.notna()
        if int(m.sum()) < 30:
            continue
        try:
            r = float(pd.Series(x[m]).corr(pd.Series(s[m]), method="spearman"))
        except Exception:
            continue
        if not np.isfinite(r):
            continue
        rows.append((c, r, float(abs(r)), int(m.sum())))
    rows.sort(key=lambda t: t[2], reverse=True)
    corr_df = pd.DataFrame(rows, columns=["feature", "spearman_r", "abs_r", "n"])
    corr_df.to_csv(diag / "score_feature_correlations.csv", index=False)

    keep_cols = [c for c in ["source_id", score_col, "ra", "dec", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"] if c in df.columns]
    if not keep_cols:
        keep_cols = [score_col]
    top = df.copy()
    top[score_col] = s
    top = top.sort_values(score_col, ascending=False).head(200)
    top[keep_cols].to_csv(diag / "top_outliers.csv", index=False)

    top_feats = corr_df.head(3)["feature"].tolist() if not corr_df.empty else []
    for k, feat in enumerate(top_feats, start=1):
        out_png = diag / f"scatter_score_vs_{feat}.png"
        out_png_generic = diag / f"scatter_score_vs_feature_{k}.png"
        x = pd.to_numeric(df[feat], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        y = s.to_numpy(float)
        fig = plt.figure(figsize=(9.5, 6.5), dpi=170)
        ax = plt.gca()
        ax.set_facecolor("#07080c")
        fig.patch.set_facecolor("#07080c")
        ax.scatter(x, y, s=10, alpha=0.55)
        ax.set_title(f"Score vs {feat}", color="white")
        ax.set_xlabel(feat, color="white")
        ax.set_ylabel(score_col, color="white")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        try:
            fig.savefig(out_png_generic, dpi=220, bbox_inches="tight")
        except Exception:
            pass
        plt.close(fig)

    for k in (1, 2, 3):
        p = diag / f"scatter_score_vs_feature_{k}.png"
        if not p.exists():
            _placeholder_png(p, "Diagnostics", "Not enough correlated features.")


def export_graph_layout_cloud(df: pd.DataFrame, G: Optional[object], out_png: Path, out_html: Path) -> None:
    """Topology-first embedding using a spring layout on the graph (if available)."""
    if G is None or not hasattr(G, "number_of_nodes") or int(G.number_of_nodes()) == 0:
        _placeholder_png(out_png, "Graph layout cloud", "Graph not available.")
        _placeholder_html(out_html, "Graph layout cloud", "Graph not available.")
        return

    try:
        import networkx as nx  # type: ignore
    except Exception:
        _placeholder_png(out_png, "Graph layout cloud", "networkx not installed.")
        _placeholder_html(out_html, "Graph layout cloud", "networkx not installed.")
        return

    nodes = list(G.nodes())
    if len(nodes) > 4000:
        rng = np.random.default_rng(42)
        nodes = rng.choice(nodes, size=4000, replace=False).tolist()
        H = G.subgraph(nodes).copy()
    else:
        H = G

    try:
        pos = nx.spring_layout(H, seed=42, iterations=60)
    except Exception as e:
        _placeholder_png(out_png, "Graph layout cloud", f"Layout failed: {e}")
        _placeholder_html(out_html, "Graph layout cloud", f"Layout failed: {e}")
        return

    score_col = "anomaly_score_norm" if "anomaly_score_norm" in df.columns else ("anomaly_score" if "anomaly_score" in df.columns else None)
    if score_col is None or "source_id" not in df.columns:
        _placeholder_png(out_png, "Graph layout cloud", "Missing score or source_id.")
        _placeholder_html(out_html, "Graph layout cloud", "Missing score or source_id.")
        return

    score = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0)
    score_map = dict(zip(df["source_id"].astype(str), score.astype(float)))

    xs: List[float] = []
    ys: List[float] = []
    cs: List[float] = []
    for node, (x, y) in pos.items():
        key = str(node)
        xs.append(float(x))
        ys.append(float(y))
        cs.append(float(score_map.get(key, 0.0)))

    xs_arr = np.array(xs, dtype=float)
    ys_arr = np.array(ys, dtype=float)
    cs_arr = np.array(cs, dtype=float)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8), dpi=170)
    ax = plt.gca()
    ax.set_facecolor("#07080c")
    fig.patch.set_facecolor("#07080c")
    ax.scatter(xs_arr, ys_arr, s=9, alpha=0.55)
    idx = np.argsort(cs_arr)[::-1]
    top = idx[: min(160, max(30, int(0.08 * len(cs_arr))))]
    ax.scatter(xs_arr[top], ys_arr[top], s=24, alpha=0.95)
    ax.set_title("Graph layout cloud (spring)", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    if not _HAS_PLOTLY:
        _placeholder_html(out_html, "Graph layout cloud", "Plotly not installed. Install requirements_viz.txt.")
        return

    try:
        fig3 = go.Figure()
        fig3.add_trace(go.Scattergl(x=xs_arr, y=ys_arr, mode="markers", marker=dict(size=5, opacity=0.55), name="nodes"))
        fig3.add_trace(go.Scattergl(x=xs_arr[top], y=ys_arr[top], mode="markers", marker=dict(size=8, opacity=0.95), name="top anomalies"))
        fig3.update_layout(title="Graph layout cloud (spring)", template="plotly_dark")
        _write_plotly_html(fig3, out_html, "Graph layout cloud")
    except Exception as e:
        _placeholder_html(out_html, "Graph layout cloud", f"Plotly export failed: {e}")

def export_dashboard(out_dir: Path) -> None:
    rel = lambda p: p.name
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>AstroGraphAnomaly — A→H Gallery</title>
<style>
  body {{ background:#07080c; color:#eaeaea; font-family: ui-sans-serif, system-ui; margin: 24px; }}
  a {{ color:#8fb6ff; }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ background:#0d1018; border:1px solid #1b2233; border-radius:14px; padding:14px; }}
  img {{ max-width: 100%; border-radius: 10px; }}
  h1,h2 {{ margin: 8px 0; }}
  .links a {{ margin-right: 14px; }}
</style>
</head>
<body>
<h1>AstroGraphAnomaly — A→H Gallery</h1>
<div class="links">
  <a href="{rel(out_dir/'02_celestial_sphere_3d.html')}">B) Celestial Sphere 3D</a>
  <a href="{rel(out_dir/'03_network_explorer.html')}">C) Network Explorer</a>
  <a href="{rel(out_dir/'08_feature_biocubes.html')}">G) BioCubes</a>
  <a href="{rel(out_dir/'10_umap_cosmic_cloud.html')}">H) UMAP (interactive)</a>
  <a href="{rel(out_dir/'12_hr_cmd_outliers.html')}">I) HR/CMD (interactive)</a>
  <a href="{rel(out_dir/'14_graph_layout_cloud.html')}">J) Graph layout (interactive)</a>
  <a href="{rel(out_dir/'diagnostics/viz_profile.json')}">Diagnostics profile</a>
</div>

<h2>Curated visuals</h2>
<div class="grid">
  <div class="card"><h3>A) Hidden Constellations</h3><img src="{rel(out_dir/'01_hidden_constellations_sky.png')}" /></div>
  <div class="card"><h3>H) UMAP Cosmic Cloud</h3><img src="{rel(out_dir/'09_umap_cosmic_cloud.png')}" /></div>
  <div class="card"><h3>D) Explainability Heatmap</h3><img src="{rel(out_dir/'04_explainability_heatmap.png')}" /></div>
  <div class="card"><h3>D) Feature Interaction</h3><img src="{rel(out_dir/'05_feature_interaction_heatmap.png')}" /></div>
  <div class="card"><h3>F) Proper Motion Trails</h3><img src="{rel(out_dir/'07_proper_motion_trails.gif')}" /></div>
  <div class="card"><h3>I) HR/CMD outliers</h3><img src="{rel(out_dir/'11_hr_cmd_outliers.png')}" /></div>
  <div class="card"><h3>J) Graph layout cloud</h3><img src="{rel(out_dir/'13_graph_layout_cloud.png')}" /></div>
</div>


<h2>Diagnostics</h2>
<div class="card">
  <p>Profile: <a href="diagnostics/viz_profile.json">viz_profile.json</a></p>
  <p>Correlations: <a href="diagnostics/score_feature_correlations.csv">score_feature_correlations.csv</a></p>
  <p>Top outliers: <a href="diagnostics/top_outliers.csv">top_outliers.csv</a></p>
  <div class="grid">
    <div class="card"><h3>Score vs feature (1)</h3><img src="diagnostics/scatter_score_vs_feature_1.png" /></div>
    <div class="card"><h3>Score vs feature (2)</h3><img src="diagnostics/scatter_score_vs_feature_2.png" /></div>
    <div class="card"><h3>Score vs feature (3)</h3><img src="diagnostics/scatter_score_vs_feature_3.png" /></div>
  </div>
</div>

</body>
</html>
"""
    (out_dir / "06_explorer_dashboard.html").write_text(html, encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate A→H visualization suite for a run directory.")
    ap.add_argument("--run-dir", required=True, help="Run directory (outputs will be written under <run-dir>/viz_a_to_h)")
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--graph", default="", help="Path to graph graphml (optional). If empty, auto-detect in run-dir.")
    ap.add_argument("--explain", default="", help="Path to explanations.jsonl (optional)")
    ap.add_argument("--allow-missing-viz-deps", action="store_true", help="If set, generate placeholders instead of failing when Plotly/PyVis/ImageIO are missing.")

    # Multi-color incoherence settings
    ap.add_argument(
        "--color-mode",
        default="auto",
        choices=["auto", "score", "dominant_phi", "rgb_phi"],
        help="Coloring mode for Plotly/PyVis visuals. auto uses dominant_phi if phi_* exists else score.",
    )
    ap.add_argument("--phi-prefix", default="phi_", help="Prefix for incoherence columns (default: phi_)")
    ap.add_argument("--phi-weights", default="", help='Weights for incoherence columns, e.g. "graph=2.0,lof=1.0"')
    ap.add_argument("--rgb-phis", default="", help='For rgb_phi: 3 channel names (no prefix), e.g. "graph,lof,ocsvm"')
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    scored = Path(args.scored)
    explain = Path(args.explain) if args.explain else None

    out_dir = run_dir / "viz_a_to_h"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(scored)
    df = ensure_core(df)

    # Apply multi-color incoherence model (if phi_* exists or explicitly requested)
    df = _apply_viz_colors(
        df,
        color_mode=args.color_mode,
        phi_prefix=args.phi_prefix,
        phi_weights_spec=args.phi_weights,
        rgb_phis_spec=args.rgb_phis,
    )

    graph_path = pick_graph_path(run_dir, args.graph)
    G = load_graph(graph_path) if graph_path is not None else None

    if G is not None and "source_id" in df.columns:
        comm = community_labels(G)
        df["community_id"] = df["source_id"].astype(str).map(comm).fillna(-1).astype(int)

    profile = compute_viz_profile(df, G, graph_path, args)
    _safe_call(
        "A) Hidden Constellations",
        [(out_dir / "01_hidden_constellations_sky.png", "png")],
        lambda: plot_hidden_constellations(df, G, out_dir / "01_hidden_constellations_sky.png"),
        profile=profile,
    )
    _safe_call(
        "B) Celestial Sphere 3D",
        [(out_dir / "02_celestial_sphere_3d.html", "html")],
        lambda: export_celestial_sphere(df, out_dir / "02_celestial_sphere_3d.html"),
        profile=profile,
    )
    _safe_call(
        "C) Network Explorer",
        [(out_dir / "03_network_explorer.html", "html")],
        lambda: export_network_explorer(df, G, out_dir / "03_network_explorer.html"),
        profile=profile,
    )
    _safe_call(
        "J) Graph layout cloud",
        [(out_dir / "13_graph_layout_cloud.png", "png"), (out_dir / "14_graph_layout_cloud.html", "html")],
        lambda: export_graph_layout_cloud(df, G, out_dir / "13_graph_layout_cloud.png", out_dir / "14_graph_layout_cloud.html"),
        profile=profile,
    )
    _safe_call(
        "D) Explainability Heatmap",
        [(out_dir / "04_explainability_heatmap.png", "png")],
        lambda: plot_explainability_heatmap(df, explain, out_dir / "04_explainability_heatmap.png", top_n=40),
        profile=profile,
    )
    _safe_call(
        "D) Feature Interaction Heatmap",
        [(out_dir / "05_feature_interaction_heatmap.png", "png")],
        lambda: plot_feature_interaction_heatmap(df, out_dir / "05_feature_interaction_heatmap.png"),
        profile=profile,
    )
    _safe_call(
        "F) Proper Motion Trails",
        [(out_dir / "07_proper_motion_trails.gif", "gif")],
        lambda: export_proper_motion_trails(df, out_dir / "07_proper_motion_trails.gif", top_k=30, frames=24),
        profile=profile,
    )
    _safe_call(
        "G) Feature BioCubes",
        [(out_dir / "08_feature_biocubes.html", "html")],
        lambda: export_feature_biocubes(df, out_dir / "08_feature_biocubes.html"),
        profile=profile,
    )
    _safe_call(
        "H) UMAP cosmic cloud",
        [(out_dir / "09_umap_cosmic_cloud.png", "png"), (out_dir / "10_umap_cosmic_cloud.html", "html")],
        lambda: export_umap(df, out_dir / "09_umap_cosmic_cloud.png", out_dir / "10_umap_cosmic_cloud.html"),
        profile=profile,
    )
    _safe_call(
        "I) HR/CMD outliers",
        [(out_dir / "11_hr_cmd_outliers.png", "png"), (out_dir / "12_hr_cmd_outliers.html", "html")],
        lambda: plot_hr_cmd_outliers(df, out_dir / "11_hr_cmd_outliers.png", out_dir / "12_hr_cmd_outliers.html"),
        profile=profile,
    )
    _safe_call(
        "Diagnostics",
        [
            (out_dir / "diagnostics" / "scatter_score_vs_feature_1.png", "png"),
            (out_dir / "diagnostics" / "scatter_score_vs_feature_2.png", "png"),
            (out_dir / "diagnostics" / "scatter_score_vs_feature_3.png", "png"),
        ],
        lambda: export_diagnostics(df, out_dir, profile),
        profile=profile,
    )
    _safe_call(
        "E) Explorer Dashboard",
        [(out_dir / "06_explorer_dashboard.html", "html")],
        lambda: export_dashboard(out_dir),
        profile=profile,
    )

    print("OK: wrote A→H (+HR/CMD) gallery to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


    raise SystemExit(main())