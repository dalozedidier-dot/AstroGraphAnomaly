#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — A→H Visualization Suite (robust + multicolor)
----------------------------------------------------------------

Generates a compact "gallery" of visual artifacts from a run directory:

A) Sky map anomalies (Hidden Constellations-style)
B) Celestial sphere 3D (interactive Plotly HTML)
C) Interactive network explorer (PyVis HTML)
D) Explainability heatmap (LIME jsonl if available, else robust z-scores)
E) Dashboard (HTML index)
F) Proper motion trails (GIF)
G) Feature BioCubes (Plotly 3D)
H) UMAP cosmic cloud (PNG + Plotly HTML)
I) HR/CMD outliers (PNG + Plotly HTML)

Design goals:
- Workflow-first: standalone script, works in CI.
- Never hard-fail the whole suite if one visualization fails.
- Optional multi-color encoding by "incoherence" constraints using phi_* (or score_*) columns.

Inputs:
- scored.csv (required)
- graph_full.graphml or graph_union.graphml (optional)
- explanations.jsonl (optional)

Outputs:
- <run-dir>/viz_a_to_h/01.. (files) + 06_explorer_dashboard.html
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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
    from plotly.offline import plot as plotly_plot  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    go = None
    plotly_plot = None
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


# -----------------------------
# Utilities
# -----------------------------

def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad < 1e-12:
        s = np.nanstd(x)
        return (x - med) / (s + 1e-12)
    return (x - med) / (1.4826 * mad + 1e-12)


def robust_unit_interval(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    lo = np.nanpercentile(x, 5)
    hi = np.nanpercentile(x, 95)
    if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-12:
        y = x - np.nanmin(x)
        den = np.nanmax(y) + 1e-12
        return np.clip(y / den, 0.0, 1.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _parse_kv_floats(s: str) -> Dict[str, float]:
    """
    Parse: "graph=2.0,lof=1,ocsvm=0.8" -> dict
    """
    out: Dict[str, float] = {}
    if not s:
        return out
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        k = k.strip()
        try:
            out[k] = float(v.strip())
        except Exception:
            continue
    return out


def _first_present(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept a few common variant names and map to canonical columns used by the suite:
    ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp, parallax, distance
    """
    df = df.copy()

    canon_map = {
        "ra": ["ra", "ra_deg", "ra_degree"],
        "dec": ["dec", "dec_deg", "dec_degree"],
        "pmra": ["pmra", "pm_ra", "pmra_masyr", "pmra_mas_per_yr"],
        "pmdec": ["pmdec", "pm_dec", "pmdec_masyr", "pmdec_mas_per_yr"],
        "phot_g_mean_mag": ["phot_g_mean_mag", "g_mag", "phot_g_mag"],
        "bp_rp": ["bp_rp", "bp_rp_color", "bp_minus_rp"],
        "parallax": ["parallax", "plx", "parallax_mas"],
        "distance": ["distance", "distance_pc", "dist_pc"],
        "source_id": ["source_id", "id", "gaia_id"],
    }

    for canon, cands in canon_map.items():
        if canon in df.columns:
            continue
        src = _first_present(df, cands)
        if src is not None:
            df[canon] = df[src]

    # If bp_rp is missing but bp and rp exist, try compute.
    if "bp_rp" not in df.columns:
        bp = _first_present(df, ["phot_bp_mean_mag", "bp_mag", "phot_bp_mag"])
        rp = _first_present(df, ["phot_rp_mean_mag", "rp_mag", "phot_rp_mag"])
        if bp is not None and rp is not None:
            df["bp_rp"] = pd.to_numeric(df[bp], errors="coerce") - pd.to_numeric(df[rp], errors="coerce")

    return df


# -----------------------------
# Core derived fields
# -----------------------------

def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = _canonicalize_columns(df)

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

    # distance helper
    if "distance" not in df.columns and "parallax" in df.columns:
        par = pd.to_numeric(df["parallax"], errors="coerce")
        df["distance"] = (1000.0 / par).replace([np.inf, -np.inf], np.nan)

    return df


# -----------------------------
# Graph helpers
# -----------------------------

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
    nodes = [str(n) for n in G.nodes()]
    if len(nodes) <= max_nodes:
        edges = [(str(u), str(v)) for u, v in G.edges()]
        return nodes, edges

    d = df.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d[d["source_id"].astype(str).isin(nodes)]
    d = d.sort_values("anomaly_score_hi", ascending=False)

    top_keep = d.head(min(250, len(d)))["source_id"].astype(str).tolist()

    comm = {}
    if "community_id" in d.columns:
        comm = dict(zip(d["source_id"].astype(str).tolist(), d["community_id"].to_numpy(int)))

    keep = set(top_keep)
    if comm:
        for cid in sorted(set(comm.values())):
            pool = d[(d["community_id"] == cid) & (d["anomaly_label"].astype(int) != -1)]["source_id"].astype(str).tolist()
            if not pool:
                continue
            k = min(25, len(pool))
            keep.update(rng.choice(pool, size=k, replace=False).tolist())

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


# -----------------------------
# Color encoding (score vs constraints)
# -----------------------------

_PALETTE = [
    "#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2",
    "#EECA3B", "#B279A2", "#FF9DA6", "#9D755D", "#BAB0AC",
    "#1F77B4", "#D62728",
]


def _phi_columns(df: pd.DataFrame, phi_prefix: str) -> List[str]:
    cols = []
    for c in df.columns:
        if c.startswith(phi_prefix) and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return sorted(cols)


def annotate_viz_colors(
    df: pd.DataFrame,
    color_mode: str,
    phi_prefix: str,
    phi_weights: Dict[str, float],
    rgb_phis: Sequence[str],
) -> pd.DataFrame:
    """
    Adds columns:
    - viz_color_type: continuous | categorical | rgb
    - viz_color_value: float (continuous) OR "rgb(...)" (categorical/rgb)
    - viz_group: constraint name (categorical)
    - viz_r01,viz_g01,viz_b01 for rgb mode
    - viz_constraints_html: short breakdown for hover
    """
    df = df.copy()
    df["viz_color_type"] = "continuous"
    df["viz_color_value"] = df["anomaly_score_norm"].to_numpy(float)

    phi_cols = _phi_columns(df, phi_prefix)

    if color_mode == "score" or not phi_cols:
        df["viz_constraints_html"] = ""
        return df

    # build phi matrix
    keys = [c[len(phi_prefix):] for c in phi_cols]
    Phi = np.zeros((len(df), len(phi_cols)), dtype=float)
    for j, c in enumerate(phi_cols):
        x = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        # ensure in [0,1] even if user provides raw-ish numbers
        Phi[:, j] = robust_unit_interval(x)

    w = np.array([float(phi_weights.get(k, 1.0)) for k in keys], dtype=float)
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    WPhi = Phi * w[None, :]

    # constraints breakdown for hover
    topk = min(3, len(keys))
    idx_sorted = np.argsort(-WPhi, axis=1)[:, :topk]
    parts = []
    for i in range(len(df)):
        lst = []
        for j in idx_sorted[i]:
            lst.append(f"{keys[j]}: {WPhi[i, j]:.3f}")
        parts.append("constraints: " + " | ".join(lst))
    df["viz_constraints_html"] = parts

    if color_mode == "dominant_phi":
        dom = np.argmax(WPhi, axis=1)
        groups = [keys[i] for i in dom]
        df["viz_group"] = groups
        # stable mapping (sorted keys -> palette)
        uniq = sorted(set(keys))
        cmap = {k: _PALETTE[i % len(_PALETTE)] for i, k in enumerate(uniq)}
        df["viz_color_value"] = [cmap[g] for g in groups]
        df["viz_color_type"] = "categorical"
        return df

    if color_mode == "rgb_phi":
        # pick three constraints
        if len(rgb_phis) != 3:
            # fallback: choose first three keys
            rgb_keys = keys[:3]
        else:
            rgb_keys = list(rgb_phis)
        # map keys -> col index
        key_to_j = {k: j for j, k in enumerate(keys)}
        rr = Phi[:, key_to_j.get(rgb_keys[0], 0)]
        gg = Phi[:, key_to_j.get(rgb_keys[1], 1 if len(keys) > 1 else 0)]
        bb = Phi[:, key_to_j.get(rgb_keys[2], 2 if len(keys) > 2 else 0)]
        df["viz_r01"] = rr
        df["viz_g01"] = gg
        df["viz_b01"] = bb
        df["viz_color_value"] = [f"rgb({int(255*r)},{int(255*g)},{int(255*b)})" for r, g, b in zip(rr, gg, bb)]
        df["viz_color_type"] = "rgb"
        return df

    # unknown mode -> continuous
    df["viz_constraints_html"] = ""
    return df


def _hover_text(df: pd.DataFrame, extra: Optional[Sequence[str]] = None) -> Optional[pd.Series]:
    cols = ["source_id", "anomaly_score_hi", "anomaly_score_norm"]
    if extra:
        cols.extend([c for c in extra if c in df.columns and c not in cols])
    base = None
    if cols:
        base = df[cols].astype(str).agg("<br>".join, axis=1)
    if "viz_constraints_html" in df.columns:
        if base is None:
            base = df["viz_constraints_html"].astype(str)
        else:
            base = base + "<br>" + df["viz_constraints_html"].astype(str)
    return base


# -----------------------------
# Rendering helpers
# -----------------------------

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


def _glow_scatter(x: np.ndarray, y: np.ndarray, score: np.ndarray, s: int, ax) -> None:
    # light halo + core
    ax.scatter(x, y, s=s, alpha=0.06 + 0.12*score, linewidths=0)
    ax.scatter(x, y, s=max(1, int(0.45*s)), alpha=0.10 + 0.25*score, linewidths=0)


def _write_error_png(out_png: Path, title: str, err: Exception) -> None:
    fig = plt.figure(figsize=(10, 3))
    plt.axis("off")
    msg = f"{title}\n\n{type(err).__name__}: {err}"
    plt.text(0.02, 0.5, msg, va="center")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _write_error_html(out_html: Path, title: str, err: Exception) -> None:
    html = f"""<!doctype html><meta charset="utf-8">
<title>{title}</title>
<body style="font-family:system-ui;background:#0b0d12;color:#eaeaea;padding:18px">
<h2>{title}</h2>
<pre>{type(err).__name__}: {err}</pre>
</body>"""
    out_html.write_text(html, encoding="utf-8")


def _safe_call(fn, out_path: Path, title: str) -> None:
    try:
        fn()
    except Exception as e:
        # best-effort placeholder
        if out_path.suffix.lower() == ".html":
            _write_error_html(out_path, title, e)
        elif out_path.suffix.lower() == ".png":
            _write_error_png(out_path, title, e)
        elif out_path.suffix.lower() == ".gif":
            # minimal 1-frame gif if possible
            side = out_path.with_suffix(".gif.error.txt")
            side.write_text(f"{title}\n{type(e).__name__}: {e}", encoding="utf-8")
            if _HAS_IMAGEIO:
                img = np.zeros((180, 420, 3), dtype=np.uint8)
                imageio.mimsave(out_path, [img], duration=0.5)
        else:
            out_path.write_text(f"{title}\n{type(e).__name__}: {e}", encoding="utf-8")


# -----------------------------
# A) Sky map anomalies
# -----------------------------

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
    xedges = np.linspace(np.nanmin(ra), np.nanmax(ra), bins + 1)
    yedges = np.linspace(np.nanmin(dec), np.nanmax(dec), bins + 1)
    H, _, _ = np.histogram2d(ra, dec, bins=[xedges, yedges])
    H = _blur2d(H.T, sigma=1.6)

    fig = plt.figure(figsize=(14, 8))
    ax = plt.gca()
    ax.imshow(
        H,
        origin="lower",
        aspect="auto",
        extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        alpha=0.95,
    )

    # Keep the glow base for density.
    _glow_scatter(ra, dec, score, s=8, ax=ax)

    # Optional categorical overlay for top anomalies
    if df.get("viz_color_type", "continuous").iloc[0] == "categorical" and "viz_color_value" in df.columns:
        top = df.sort_values("anomaly_score_hi", ascending=False).head(min(180, len(df)))
        ax.scatter(
            pd.to_numeric(top["ra"], errors="coerce"),
            pd.to_numeric(top["dec"], errors="coerce"),
            s=20,
            alpha=0.85,
            c=top["viz_color_value"].tolist(),
            linewidths=0.0,
        )

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
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            dx, dy = (x2 - x1), (y2 - y1)
            norm = math.hypot(dx, dy) + 1e-9
            px, py = (-dy / norm, dx / norm)
            offset = 0.08 * norm
            cx, cy = mx + px * offset, my + py * offset
            t = np.linspace(0, 1, 20)
            bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t**2 * x2
            by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t**2 * y2
            w = 0.6 if (su in top_ids or sv in top_ids) else 0.35
            a = 0.22 if (su in top_ids or sv in top_ids) else 0.12
            ax.plot(bx, by, linewidth=w, alpha=a)
            kept += 1

    ax.set_title("Sky map anomalies (Hidden Constellations style)")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.grid(alpha=0.10)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# B) Celestial sphere 3D (Plotly)
# -----------------------------

def export_celestial_sphere(df: pd.DataFrame, out_html: Path) -> None:
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    if "ra" not in df.columns or "dec" not in df.columns:
        out_html.write_text("Missing ra/dec in scored.csv", encoding="utf-8")
        return

    ra = np.deg2rad(pd.to_numeric(df["ra"], errors="coerce").fillna(0.0).to_numpy(float))
    dec = np.deg2rad(pd.to_numeric(df["dec"], errors="coerce").fillna(0.0).to_numpy(float))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    score = df["anomaly_score_norm"].to_numpy(float)
    size = 3 + 10 * score

    hover = _hover_text(df, extra=["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"])

    ctype = str(df.get("viz_color_type", pd.Series(["continuous"])).iloc[0])

    if ctype == "categorical" and "viz_group" in df.columns and "viz_color_value" in df.columns:
        # one trace per group to get a legend
        fig = go.Figure()
        groups = df["viz_group"].astype(str).fillna("unknown")
        for g in sorted(groups.unique()):
            m = (groups == g).to_numpy(bool)
            fig.add_trace(go.Scatter3d(
                x=x[m], y=y[m], z=z[m],
                mode="markers",
                name=g,
                marker=dict(size=size[m], color=df.loc[m, "viz_color_value"].iloc[0], opacity=0.90),
                text=None if hover is None else hover[m],
                hoverinfo="text",
            ))
        title = "Celestial Sphere 3D (dominant incoherence constraint)"
    elif ctype == "rgb" and "viz_color_value" in df.columns:
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=size, color=df["viz_color_value"].tolist(), opacity=0.90),
            text=hover,
            hoverinfo="text",
        )])
        title = "Celestial Sphere 3D (RGB constraints blend)"
    else:
        fig = go.Figure(data=[go.Scatter3d(
            x=x, y=y, z=z, mode="markers",
            marker=dict(size=size, color=score, colorscale="Viridis", opacity=0.88, colorbar=dict(title="anomaly")),
            text=hover,
            hoverinfo="text",
        )])
        title = "Celestial Sphere 3D (anomaly score)"

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(itemsizing="constant"),
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


# -----------------------------
# C) Network explorer (PyVis)
# -----------------------------

def export_network_explorer(df: pd.DataFrame, G_opt, out_html: Path, max_nodes: int = 1200) -> None:
    if not _HAS_PYVIS:
        out_html.write_text("PyVis not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    if G_opt is None:
        out_html.write_text("No graph provided (graph_full/union.graphml).", encoding="utf-8")
        return

    G = G_opt
    nodes_keep, edges_keep = sample_subgraph(G, df, max_nodes=max_nodes)

    d = df.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d.set_index("source_id", drop=False)

    net = Network(height="820px", width="100%", bgcolor="#05060a", font_color="#e8e8e8", directed=False)
    net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=110, spring_strength=0.08, damping=0.4)

    for sid in nodes_keep:
        row = d.loc[sid] if sid in d.index else None
        sc = float(row["anomaly_score_norm"]) if row is not None and "anomaly_score_norm" in row else 0.2
        size = 10 + 30 * sc
        label = sid if sc > 0.92 else ""

        title_parts = []
        if row is not None:
            cols = ["source_id", "anomaly_score_hi", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe", "degree", "kcore", "betweenness"]
            for c in cols:
                if c in row.index:
                    title_parts.append(f"{c}: {row[c]}")
            if "viz_constraints_html" in row.index:
                title_parts.append(str(row["viz_constraints_html"]))
        title = "<br>".join(title_parts) if title_parts else sid

        if row is not None and "viz_color_type" in row.index and row["viz_color_type"] in ("categorical", "rgb") and "viz_color_value" in row.index:
            color = str(row["viz_color_value"])
        else:
            r = int(40 + 215 * sc)
            g = int(80 + 150 * sc)
            b = int(220 - 160 * sc)
            color = f"rgb({r},{g},{b})"

        net.add_node(sid, label=label, title=title, value=size, color=color)

    for u, v in edges_keep:
        net.add_edge(u, v, value=1)

    net.save_graph(str(out_html))


# -----------------------------
# D) Explainability heatmap
# -----------------------------

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
                            try:
                                pairs.append((str(it[0]), float(it[1])))
                            except Exception:
                                continue
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
        cols = [c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "distance", "ruwe", "degree", "kcore", "betweenness"] if c in num_cols]
        cols = cols[:12] if cols else num_cols[:12]
        if not cols:
            fig = plt.figure(figsize=(10, 3))
            plt.axis("off")
            plt.text(0.5, 0.5, "No numeric columns for fallback explainability.", ha="center", va="center")
            fig.savefig(out_png, dpi=240, bbox_inches="tight")
            plt.close(fig)
            return
        X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        for j in range(X.shape[1]):
            X[:, j] = robust_z(X[:, j])
        title = "Explainability heatmap (fallback: robust z-scores)"
        data = X
        ylabels = d["source_id"].astype(str).tolist()
        xlabels = cols

    fig = plt.figure(figsize=(12, 7))
    ax = plt.gca()
    im = ax.imshow(data, aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, fontsize=8, rotation=45, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.026, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# D2) Feature interaction heatmap (simple)
# -----------------------------

def plot_feature_interaction_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "distance", "ruwe", "degree", "kcore", "betweenness"] if c in num_cols]
    cols = preferred[:10] if len(preferred) >= 4 else num_cols[:10]
    if len(cols) < 2:
        fig = plt.figure(figsize=(10, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "Not enough numeric columns for interaction heatmap.", ha="center", va="center")
        fig.savefig(out_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    C = np.corrcoef(X, rowvar=False)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(C, aspect="auto", vmin=-1, vmax=1)
    ax.set_title("Feature interaction heatmap (correlation)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# F) Proper motion trails GIF
# -----------------------------

def _fig_to_rgb(fig) -> np.ndarray:
    """
    Matplotlib compatibility: newer versions may not expose tostring_rgb().
    Returns RGB uint8 image array [H,W,3].
    """
    canvas = fig.canvas
    # render
    canvas.draw()
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba())
        if buf.ndim == 3 and buf.shape[-1] == 4:
            return np.ascontiguousarray(buf[..., :3])
    if hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        arr = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return arr.reshape((h, w, 3))
    if hasattr(canvas, "tostring_argb"):
        w, h = canvas.get_width_height()
        arr = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape((h, w, 4))
        return np.ascontiguousarray(arr[..., 1:4])
    raise RuntimeError("Matplotlib canvas: no RGB buffer method available.")


def export_proper_motion_trails(df: pd.DataFrame, out_gif: Path, top_k: int = 30, frames: int = 24) -> None:
    if not _HAS_IMAGEIO:
        out_gif.write_text("imageio not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    needed = {"ra", "dec", "pmra", "pmdec"}
    if not needed.issubset(set(df.columns)):
        out_gif.write_text("Missing ra/dec/pmra/pmdec for trails.", encoding="utf-8")
        return

    d = df.sort_values("anomaly_score_hi", ascending=False).head(min(top_k, len(df))).copy()
    ra0 = pd.to_numeric(d["ra"], errors="coerce").fillna(0.0).to_numpy(float)
    dec0 = pd.to_numeric(d["dec"], errors="coerce").fillna(0.0).to_numpy(float)
    pmra = pd.to_numeric(d["pmra"], errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(d["pmdec"], errors="coerce").fillna(0.0).to_numpy(float)
    score = d["anomaly_score_norm"].to_numpy(float)

    mas2deg = 1.0 / 3.6e6
    cosd = np.cos(np.deg2rad(np.clip(dec0, -89.9, 89.9)))
    dra_deg_per_yr = (pmra * mas2deg) / np.maximum(cosd, 1e-3)
    ddec_deg_per_yr = pmdec * mas2deg

    T = 8.0
    ts = np.linspace(0.0, T, frames)

    imgs = []
    for t in ts:
        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.scatter(ra0, dec0, s=20, alpha=0.18)
        steps = 20
        tt = np.linspace(max(0.0, t - 2.0), t, steps)
        for i in range(len(ra0)):
            ra_tr = ra0[i] + dra_deg_per_yr[i] * tt
            dec_tr = dec0[i] + ddec_deg_per_yr[i] * tt
            ax.plot(ra_tr, dec_tr, alpha=0.35 + 0.4 * score[i], linewidth=1.0 + 1.2 * score[i])

        ra_t = ra0 + dra_deg_per_yr * t
        dec_t = dec0 + ddec_deg_per_yr * t
        ax.scatter(ra_t, dec_t, s=40 + 140 * score, alpha=0.85)

        ax.set_title("Proper motion trails (Top anomalies)")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.grid(alpha=0.15)

        imgs.append(_fig_to_rgb(fig))
        plt.close(fig)

    imageio.mimsave(out_gif, imgs, duration=0.11)


# -----------------------------
# G) Feature BioCubes (Plotly)
# -----------------------------

def export_feature_biocubes(df: pd.DataFrame, out_html: Path) -> None:
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return

    y = df["anomaly_label"].to_numpy(int) if "anomaly_label" in df.columns else np.ones(len(df), dtype=int)
    an = df[y == -1]
    no = df[y != -1]

    candidates = [
        c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "distance", "ruwe", "degree", "kcore", "betweenness"]
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    feats = candidates[:8] if len(candidates) >= 5 else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:6]
    if not feats:
        out_html.write_text("No numeric features found.", encoding="utf-8")
        return

    def stats(d: pd.DataFrame, col: str):
        x = pd.to_numeric(d[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if x.size == 0:
            return 0.0, 0.0, 0.0
        q1, med, q3 = np.percentile(x, [25, 50, 75])
        return float(q1), float(med), float(q3)

    meshes = []
    for i, f in enumerate(feats):
        for gi, (name, dset) in enumerate([("Normal", no), ("Anomalous", an)]):
            q1, med, q3 = stats(dset, f)
            x0, x1 = i - 0.35, i + 0.35
            y0, y1 = gi - 0.35, gi + 0.35
            z0, z1 = q1, q3

            vx = [x0, x1, x1, x0, x0, x1, x1, x0]
            vy = [y0, y0, y1, y1, y0, y0, y1, y1]
            vz = [z0, z0, z0, z0, z1, z1, z1, z1]

            I = [0, 0, 0, 4, 4, 4, 0, 0, 1, 1, 2, 2]
            J = [1, 2, 3, 5, 6, 7, 4, 5, 2, 6, 3, 7]
            K = [2, 3, 1, 6, 7, 5, 5, 6, 6, 5, 7, 6]

            meshes.append(go.Mesh3d(x=vx, y=vy, z=vz, i=I, j=J, k=K, opacity=0.25))
            meshes.append(go.Scatter3d(x=[i], y=[gi], z=[med], mode="markers", marker=dict(size=5), name=f"{f} {name}"))

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title="Feature BioCubes (IQR boxes + median markers)",
        scene=dict(
            xaxis=dict(title="feature index", tickmode="array", tickvals=list(range(len(feats))), ticktext=feats),
            yaxis=dict(title="group", tickmode="array", tickvals=[0, 1], ticktext=["Normal", "Anomalous"]),
            zaxis=dict(title="value"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False,
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


# -----------------------------
# H) UMAP cosmic cloud
# -----------------------------

def export_umap(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    ignore = {"source_id"}
    num_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "distance", "ruwe", "degree", "kcore", "betweenness"] if c in num_cols]
    cols = preferred if len(preferred) >= 5 else num_cols[:8]
    if len(cols) < 2:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Not enough numeric columns for UMAP", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        out_html.write_text("Not enough numeric columns for UMAP", encoding="utf-8")
        return

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.18, random_state=42)
        emb = reducer.fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        emb = U[:, :2] * S[:2]

    score = df["anomaly_score_norm"].to_numpy(float)
    ctype = str(df.get("viz_color_type", pd.Series(["continuous"])).iloc[0])

    # PNG
    fig = plt.figure(figsize=(10, 8))
    if ctype == "categorical" and "viz_group" in df.columns and "viz_color_value" in df.columns:
        for g in sorted(df["viz_group"].astype(str).unique()):
            m = (df["viz_group"].astype(str) == g).to_numpy(bool)
            plt.scatter(emb[m, 0], emb[m, 1], s=18, alpha=0.80, c=df.loc[m, "viz_color_value"].tolist(), label=g)
        plt.legend(loc="best", fontsize=8, frameon=False)
    elif ctype == "rgb" and {"viz_r01", "viz_g01", "viz_b01"}.issubset(df.columns):
        rgb = df[["viz_r01", "viz_g01", "viz_b01"]].to_numpy(float)
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.80, c=rgb)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=score)
        plt.colorbar(label="anomaly_score_norm")

    plt.title("Cosmic cloud embedding (UMAP if available)")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    # HTML
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return

    hover = _hover_text(df, extra=["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"])

    if ctype == "categorical" and "viz_group" in df.columns and "viz_color_value" in df.columns:
        fig2 = go.Figure()
        for g in sorted(df["viz_group"].astype(str).unique()):
            m = (df["viz_group"].astype(str) == g).to_numpy(bool)
            fig2.add_trace(go.Scattergl(
                x=emb[m, 0], y=emb[m, 1], mode="markers",
                name=g,
                marker=dict(size=6 + 9 * score[m], color=df.loc[m, "viz_color_value"].iloc[0], opacity=0.88),
                text=None if hover is None else hover[m],
                hoverinfo="text",
            ))
        title = "UMAP cosmic cloud (dominant incoherence constraint)"
    elif ctype == "rgb" and "viz_color_value" in df.columns:
        fig2 = go.Figure(data=[go.Scattergl(
            x=emb[:, 0], y=emb[:, 1], mode="markers",
            marker=dict(size=6 + 9 * score, color=df["viz_color_value"].tolist(), opacity=0.88),
            text=hover,
            hoverinfo="text",
        )])
        title = "UMAP cosmic cloud (RGB constraints blend)"
    else:
        fig2 = go.Figure(data=[go.Scattergl(
            x=emb[:, 0], y=emb[:, 1], mode="markers",
            marker=dict(size=6 + 9 * score, color=score, colorscale="Viridis", opacity=0.88, colorbar=dict(title="anomaly")),
            text=hover,
            hoverinfo="text",
        )])
        title = "UMAP cosmic cloud (anomaly score)"

    fig2.update_layout(title=title, margin=dict(l=0, r=0, b=0, t=40))
    plotly_plot(fig2, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


# -----------------------------
# I) HR/CMD outliers
# -----------------------------

def plot_hr_cmd(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    # Need color and magnitude
    if "phot_g_mean_mag" not in df.columns or "bp_rp" not in df.columns:
        fig = plt.figure(figsize=(10, 3))
        plt.axis("off")
        plt.text(0.5, 0.5, "Missing phot_g_mean_mag or bp_rp for HR/CMD.", ha="center", va="center")
        fig.savefig(out_png, dpi=240, bbox_inches="tight")
        plt.close(fig)
        out_html.write_text("Missing phot_g_mean_mag or bp_rp for HR/CMD.", encoding="utf-8")
        return

    bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    gmag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)

    # absolute magnitude if possible
    y_is_abs = False
    if "distance" in df.columns:
        dist = pd.to_numeric(df["distance"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        ok = np.isfinite(dist) & (dist > 0)
        if np.any(ok):
            M = np.full_like(gmag, np.nan, dtype=float)
            M[ok] = gmag[ok] + 5.0 - 5.0 * np.log10(dist[ok])
            y = np.where(np.isfinite(M), M, gmag)
            y_is_abs = True
        else:
            y = gmag
    elif "parallax" in df.columns:
        plx = pd.to_numeric(df["parallax"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
        ok = np.isfinite(plx) & (plx > 0)
        if np.any(ok):
            dist = 1000.0 / plx
            M = np.full_like(gmag, np.nan, dtype=float)
            M[ok] = gmag[ok] + 5.0 - 5.0 * np.log10(dist[ok])
            y = np.where(np.isfinite(M), M, gmag)
            y_is_abs = True
        else:
            y = gmag
    else:
        y = gmag

    score = df["anomaly_score_norm"].to_numpy(float)
    ctype = str(df.get("viz_color_type", pd.Series(["continuous"])).iloc[0])

    # PNG
    fig = plt.figure(figsize=(10, 8))
    if ctype == "categorical" and "viz_group" in df.columns and "viz_color_value" in df.columns:
        for g in sorted(df["viz_group"].astype(str).unique()):
            m = (df["viz_group"].astype(str) == g).to_numpy(bool)
            plt.scatter(bp_rp[m], y[m], s=16, alpha=0.80, c=df.loc[m, "viz_color_value"].tolist(), label=g)
        plt.legend(loc="best", fontsize=8, frameon=False)
    elif ctype == "rgb" and {"viz_r01", "viz_g01", "viz_b01"}.issubset(df.columns):
        rgb = df[["viz_r01", "viz_g01", "viz_b01"]].to_numpy(float)
        plt.scatter(bp_rp, y, s=16, alpha=0.80, c=rgb)
    else:
        plt.scatter(bp_rp, y, s=16, alpha=0.80, c=score)
        plt.colorbar(label="anomaly_score_norm")

    plt.gca().invert_yaxis()
    plt.title("HR/CMD outliers" + (" (absolute M_G)" if y_is_abs else " (apparent G)"))
    plt.xlabel("BP-RP")
    plt.ylabel("M_G" if y_is_abs else "G")
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    # HTML
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return

    hover = _hover_text(df, extra=["phot_g_mean_mag", "bp_rp", "parallax", "distance", "pmra", "pmdec", "ruwe"])
    ytitle = "M_G (absolute)" if y_is_abs else "G (apparent)"

    if ctype == "categorical" and "viz_group" in df.columns and "viz_color_value" in df.columns:
        fig2 = go.Figure()
        for g in sorted(df["viz_group"].astype(str).unique()):
            m = (df["viz_group"].astype(str) == g).to_numpy(bool)
            fig2.add_trace(go.Scattergl(
                x=bp_rp[m], y=y[m], mode="markers",
                name=g,
                marker=dict(size=6 + 10 * score[m], color=df.loc[m, "viz_color_value"].iloc[0], opacity=0.88),
                text=None if hover is None else hover[m],
                hoverinfo="text",
            ))
        title = "HR/CMD outliers (dominant incoherence constraint)"
    elif ctype == "rgb" and "viz_color_value" in df.columns:
        fig2 = go.Figure(data=[go.Scattergl(
            x=bp_rp, y=y, mode="markers",
            marker=dict(size=6 + 10 * score, color=df["viz_color_value"].tolist(), opacity=0.88),
            text=hover, hoverinfo="text",
        )])
        title = "HR/CMD outliers (RGB constraints blend)"
    else:
        fig2 = go.Figure(data=[go.Scattergl(
            x=bp_rp, y=y, mode="markers",
            marker=dict(size=6 + 10 * score, color=score, colorscale="Viridis", opacity=0.88, colorbar=dict(title="anomaly")),
            text=hover, hoverinfo="text",
        )])
        title = "HR/CMD outliers (anomaly score)"

    fig2.update_layout(
        title=title,
        xaxis_title="BP-RP",
        yaxis_title=ytitle,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    plotly_plot(fig2, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


# -----------------------------
# Dashboard
# -----------------------------

def export_dashboard(out_dir: Path) -> None:
    rel = lambda p: p.name
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>AstroGraphAnomaly Gallery</title>
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
<h1>AstroGraphAnomaly Gallery</h1>

<div class="links">
  <a href="{rel(out_dir/'02_celestial_sphere_3d.html')}">B Celestial Sphere 3D</a>
  <a href="{rel(out_dir/'03_network_explorer.html')}">C Network Explorer</a>
  <a href="{rel(out_dir/'08_feature_biocubes.html')}">G BioCubes</a>
  <a href="{rel(out_dir/'10_umap_cosmic_cloud.html')}">H UMAP interactive</a>
  <a href="{rel(out_dir/'12_hr_cmd_outliers.html')}">I HR/CMD interactive</a>
</div>

<h2>Curated visuals</h2>
<div class="grid">
  <div class="card"><h3>A Sky map anomalies</h3><img src="{rel(out_dir/'01_hidden_constellations_sky.png')}" /></div>
  <div class="card"><h3>H UMAP cosmic cloud</h3><img src="{rel(out_dir/'09_umap_cosmic_cloud.png')}" /></div>
  <div class="card"><h3>I HR/CMD outliers</h3><img src="{rel(out_dir/'11_hr_cmd_outliers.png')}" /></div>
  <div class="card"><h3>D Explainability heatmap</h3><img src="{rel(out_dir/'04_explainability_heatmap.png')}" /></div>
  <div class="card"><h3>D Feature interaction</h3><img src="{rel(out_dir/'05_feature_interaction_heatmap.png')}" /></div>
  <div class="card"><h3>F Proper motion trails</h3><img src="{rel(out_dir/'07_proper_motion_trails.gif')}" /></div>
</div>

</body>
</html>
"""
    (out_dir / "06_explorer_dashboard.html").write_text(html, encoding="utf-8")


# -----------------------------
# CLI + main
# -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Generate visualization gallery from a run directory.")
    ap.add_argument("--run-dir", required=True, help="Run directory (outputs under <run-dir>/viz_a_to_h)")
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--graph", default="", help="Path to graph graphml (optional). If empty, auto-detect in run-dir.")
    ap.add_argument("--explain", default="", help="Path to explanations.jsonl (optional)")
    ap.add_argument("--max-nodes", type=int, default=1200, help="Max nodes for network explorer sampling")

    # Color modes
    ap.add_argument("--color-mode", choices=["score", "dominant_phi", "rgb_phi"], default="score",
                    help="Color encoding: score, dominant_phi (categorical), rgb_phi (blend)")
    ap.add_argument("--phi-prefix", default="phi_", help="Prefix for constraint columns (ex: phi_ or score_)")
    ap.add_argument("--phi-weights", default="", help='Weights, ex: "graph=2.0,lof=1.0,ocsvm=1.0"')
    ap.add_argument("--rgb-phis", default="", help='Three constraint keys for rgb_phi, ex: "graph,lof,ocsvm"')
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

    graph_path = pick_graph_path(run_dir, args.graph)
    G = load_graph(graph_path) if graph_path is not None else None

    if G is not None and "source_id" in df.columns:
        comm = community_labels(G)
        df["community_id"] = df["source_id"].astype(str).map(comm).fillna(-1).astype(int)

    phi_weights = _parse_kv_floats(args.phi_weights)
    rgb_phis = [p.strip() for p in args.rgb_phis.split(",") if p.strip()]
    df = annotate_viz_colors(df, args.color_mode, args.phi_prefix, phi_weights, rgb_phis)

    # Generate artifacts, never hard-fail the suite.
    _safe_call(lambda: plot_hidden_constellations(df, G, out_dir / "01_hidden_constellations_sky.png"),
               out_dir / "01_hidden_constellations_sky.png", "A Sky map anomalies")
    _safe_call(lambda: export_celestial_sphere(df, out_dir / "02_celestial_sphere_3d.html"),
               out_dir / "02_celestial_sphere_3d.html", "B Celestial sphere 3D")
    _safe_call(lambda: export_network_explorer(df, G, out_dir / "03_network_explorer.html", max_nodes=int(args.max_nodes)),
               out_dir / "03_network_explorer.html", "C Network explorer")
    _safe_call(lambda: plot_explainability_heatmap(df, explain, out_dir / "04_explainability_heatmap.png", top_n=40),
               out_dir / "04_explainability_heatmap.png", "D Explainability heatmap")
    _safe_call(lambda: plot_feature_interaction_heatmap(df, out_dir / "05_feature_interaction_heatmap.png"),
               out_dir / "05_feature_interaction_heatmap.png", "D2 Feature interaction")
    _safe_call(lambda: export_proper_motion_trails(df, out_dir / "07_proper_motion_trails.gif", top_k=30, frames=24),
               out_dir / "07_proper_motion_trails.gif", "F Proper motion trails")
    _safe_call(lambda: export_feature_biocubes(df, out_dir / "08_feature_biocubes.html"),
               out_dir / "08_feature_biocubes.html", "G Feature BioCubes")
    _safe_call(lambda: export_umap(df, out_dir / "09_umap_cosmic_cloud.png", out_dir / "10_umap_cosmic_cloud.html"),
               out_dir / "10_umap_cosmic_cloud.html", "H UMAP cosmic cloud")
    _safe_call(lambda: plot_hr_cmd(df, out_dir / "11_hr_cmd_outliers.png", out_dir / "12_hr_cmd_outliers.html"),
               out_dir / "12_hr_cmd_outliers.html", "I HR/CMD outliers")
    _safe_call(lambda: export_dashboard(out_dir),
               out_dir / "06_explorer_dashboard.html", "E Dashboard")

    print("OK: wrote gallery to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
