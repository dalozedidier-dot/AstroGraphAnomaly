#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly A to H visualization suite

Gallery outputs
A  Sky map anomalies (Hidden Constellations style)
B  Celestial sphere 3D interactive (Plotly)
C  Graph network explorer (PyVis)
D  Explainability heatmap (LIME if available, else robust z scores)
E  Simple dashboard HTML linking everything
F  Proper motion trails GIF
G  Feature biocubes (Plotly 3D)
H  UMAP cosmic cloud (PNG plus Plotly HTML)

This script is workflow first.
It reads scored.csv produced by the pipeline, plus optional graph and explanations jsonl.

Key extra feature
You can color the sphere, the network, the UMAP and the HR CMD plot using multi incoherence modes.

Color modes
score         Continuous anomaly_score_norm
dominant_phi  Categorical dominant constraint from phi_* columns
rgb_phi       RGB mix from three selected phi columns

Expected columns for multi incoherence
phi_* columns in scored.csv, normalized or not.
For example phi_isolation_forest, phi_lof, phi_ocsvm, phi_graph.

Outputs are written under: <run_dir>/viz_a_to_h/
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

try:
    import networkx as nx  # type: ignore
    _HAS_NX = True
except Exception:
    nx = None
    _HAS_NX = False


def robust_unit_interval(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=float)
    lo = float(np.nanpercentile(x, 5))
    hi = float(np.nanpercentile(x, 95))
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def robust_z(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    if not np.any(np.isfinite(x)):
        return np.zeros_like(x, dtype=float)
    med = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - med))) + 1e-12
    return (x - med) / (1.4826 * mad)


def _parse_kv_list(s: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    s = (s or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k:
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    return out


def ensure_core(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype(str)

    # Normalize naming variants for Gaia columns when possible
    rename_map = {}
    if "ra" not in df.columns:
        for cand in ["ra_deg", "raDegrees", "raJ2000", "ra_icrs"]:
            if cand in df.columns:
                rename_map[cand] = "ra"
                break
    if "dec" not in df.columns:
        for cand in ["dec_deg", "decDegrees", "decJ2000", "dec_icrs"]:
            if cand in df.columns:
                rename_map[cand] = "dec"
                break
    if rename_map:
        df = df.rename(columns=rename_map)

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


def _phi_columns(df: pd.DataFrame, phi_prefix: str) -> List[str]:
    pref = (phi_prefix or "phi_").strip()
    cols = [c for c in df.columns if c.startswith(pref)]
    return cols


def _normalize_phi_columns(df: pd.DataFrame, phi_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in phi_cols:
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        df[c] = robust_unit_interval(vals)
    return df


def _prepare_incoherence(
    df: pd.DataFrame,
    color_mode: str,
    phi_prefix: str,
    phi_weights_s: str,
    rgb_phis_s: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Prepare columns used for multicolor rendering.

    Adds:
    - incoherence_score in [0,1]
    - incoherence_dominant (string) when dominant_phi
    - incoherence_rgb (rgb string) when rgb_phi

    Returns df, and a color map for dominant categories.
    """
    df = df.copy()
    mode = (color_mode or "score").strip().lower()
    phi_prefix = (phi_prefix or "phi_").strip()

    # Default
    df["incoherence_score"] = df.get("anomaly_score_norm", 0.0)
    df["incoherence_dominant"] = ""
    df["incoherence_rgb"] = ""

    palette = [
        "#ff4d4d", "#ffd34d", "#4dff88", "#4dd2ff",
        "#b84dff", "#ff4dd2", "#7dff4d", "#4d7dff",
        "#ff8a4d", "#4dfff0",
    ]
    color_map: Dict[str, str] = {}

    if mode == "score":
        return df, color_map

    phi_cols = _phi_columns(df, phi_prefix)
    if not phi_cols:
        # No phi columns available, fallback
        return df, color_map

    df = _normalize_phi_columns(df, phi_cols)

    # Parse weights
    w_raw = _parse_kv_list(phi_weights_s)
    weights: Dict[str, float] = {}
    for c in phi_cols:
        short = c[len(phi_prefix):]
        w = w_raw.get(c, w_raw.get(short, 1.0))
        try:
            wf = float(w)
        except Exception:
            wf = 1.0
        weights[c] = max(0.0, wf)

    wsum = sum(weights.values())
    if wsum <= 1e-12:
        weights = {c: 1.0 for c in phi_cols}
        wsum = float(len(phi_cols))

    # Contributions matrix
    Phi = np.column_stack([df[c].to_numpy(float) for c in phi_cols])
    W = np.array([weights[c] for c in phi_cols], dtype=float).reshape(1, -1)
    contrib = Phi * W
    inco = contrib.sum(axis=1) / float(wsum)
    inco = np.clip(inco, 0.0, 1.0)
    df["incoherence_score"] = inco

    if mode == "dominant_phi":
        idx = np.argmax(contrib, axis=1)
        dom_cols = [phi_cols[int(i)] for i in idx.tolist()]
        dom_names = [c[len(phi_prefix):] for c in dom_cols]
        df["incoherence_dominant"] = dom_names

        uniq = sorted(set(dom_names))
        for i, name in enumerate(uniq):
            color_map[name] = palette[i % len(palette)]
        return df, color_map

    if mode == "rgb_phi":
        rgb_phis = [p.strip() for p in (rgb_phis_s or "").split(",") if p.strip()]
        rgb_cols: List[str] = []
        for p in rgb_phis:
            if p in df.columns:
                rgb_cols.append(p)
            else:
                cand = phi_prefix + p
                if cand in df.columns:
                    rgb_cols.append(cand)
        if len(rgb_cols) < 3:
            rgb_cols = phi_cols[:3] if len(phi_cols) >= 3 else phi_cols

        # Ensure 3 columns by padding with zeros if needed
        while len(rgb_cols) < 3:
            df["__phi_pad_%d" % len(rgb_cols)] = 0.0
            rgb_cols.append("__phi_pad_%d" % (len(rgb_cols) - 1))

        R = df[rgb_cols[0]].to_numpy(float)
        G = df[rgb_cols[1]].to_numpy(float)
        B = df[rgb_cols[2]].to_numpy(float)

        # Optional global intensity from incoherence_score
        I = df["incoherence_score"].to_numpy(float)
        # Blend with intensity so pale points remain visible
        R2 = np.clip(0.25 * I + 0.75 * R, 0.0, 1.0)
        G2 = np.clip(0.25 * I + 0.75 * G, 0.0, 1.0)
        B2 = np.clip(0.25 * I + 0.75 * B, 0.0, 1.0)

        rgb = []
        for r, g, b in zip(R2.tolist(), G2.tolist(), B2.tolist(), strict=False):
            rr = int(255 * float(r))
            gg = int(255 * float(g))
            bb = int(255 * float(b))
            rgb.append(f"rgb({rr},{gg},{bb})")
        df["incoherence_rgb"] = rgb
        return df, color_map

    # Unknown mode, fallback
    return df, color_map


def pick_graph_path(run_dir: Path, graph_arg: str) -> Optional[Path]:
    if graph_arg and graph_arg.strip():
        p = Path(graph_arg)
        return p if p.exists() else None
    candidates = [
        run_dir / "graph_full.graphml",
        run_dir / "graph_topk.graphml",
        run_dir / "graph_union.graphml",
        run_dir / "graph.graphml",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def load_graph(graph_path: Optional[Path]):
    if graph_path is None:
        return None
    if not _HAS_NX:
        return None
    try:
        G = nx.read_graphml(graph_path)
        if not all(isinstance(n, str) for n in G.nodes()):
            G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()}, copy=True)
        return G
    except Exception:
        return None


def community_labels(G) -> Dict[str, int]:
    if not _HAS_NX:
        return {}
    try:
        from networkx.algorithms.community import greedy_modularity_communities  # type: ignore
    except Exception:
        return {}
    try:
        comms = list(greedy_modularity_communities(G))
    except Exception:
        return {}
    labels: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            labels[str(n)] = int(i)
    return labels


def sample_subgraph(G, df: pd.DataFrame, max_nodes: int = 1200) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Keep top anomalies and some neighbors for interactive graph."""
    nodes = [str(n) for n in G.nodes()]
    if "source_id" in df.columns:
        keep = df.sort_values("incoherence_score", ascending=False)["source_id"].astype(str).head(max_nodes).tolist()
        keep_set = set(keep)
        # add neighbors for context
        for n in keep[: min(250, len(keep))]:
            if n in G:
                for nb in G.neighbors(n):
                    keep_set.add(str(nb))
        keep = list(keep_set)
    else:
        keep = nodes[:max_nodes]

    H = G.subgraph(keep)
    edges = [(str(u), str(v)) for u, v in H.edges()]
    return list(H.nodes()), edges


def _blur2d(a: np.ndarray, sigma: float) -> np.ndarray:
    if _HAS_SCIPY and gaussian_filter is not None:
        return gaussian_filter(a, sigma=sigma)
    # fallback simple box blur
    k = int(max(1, round(2 * sigma)))
    if k <= 1:
        return a
    pad = k
    ap = np.pad(a, pad, mode="reflect")
    out = np.zeros_like(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            out[i, j] = ap[i:i + 2 * pad + 1, j:j + 2 * pad + 1].mean()
    return out


def _glow_scatter(ax, x, y, s, alpha, color):
    for k, a in [(7.0, alpha * 0.06), (4.0, alpha * 0.12), (2.2, alpha * 0.22)]:
        ax.scatter(x, y, s=s * k, alpha=a, c=color, linewidths=0)


def plot_hidden_constellations(df: pd.DataFrame, G_opt, out_png: Path) -> None:
    # Coordinates from ra/dec if available, else use any 2 numeric dims
    if "ra" in df.columns and "dec" in df.columns:
        x = pd.to_numeric(df["ra"], errors="coerce").fillna(0.0).to_numpy(float)
        y = pd.to_numeric(df["dec"], errors="coerce").fillna(0.0).to_numpy(float)
    else:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            x = df[num_cols[0]].to_numpy(float)
            y = df[num_cols[1]].to_numpy(float)
        else:
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "Missing sky coordinates", ha="center", va="center")
            plt.axis("off")
            fig.savefig(out_png, dpi=260, bbox_inches="tight")
            plt.close(fig)
            return

    s = df["incoherence_score"].to_numpy(float)
    s01 = robust_unit_interval(s)

    # Density image
    W, H = 1200, 600
    xi = np.clip(((x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x) + 1e-9) * (W - 1)).astype(int), 0, W - 1)
    yi = np.clip(((y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-9) * (H - 1)).astype(int), 0, H - 1)
    dens = np.zeros((H, W), dtype=float)
    for X, Y, w in zip(xi.tolist(), yi.tolist(), (0.25 + 1.75 * s01).tolist(), strict=False):
        dens[Y, X] += w
    dens = _blur2d(dens, sigma=6.5)
    dens = dens / (dens.max() + 1e-9)

    fig = plt.figure(figsize=(14, 7), facecolor="#05060a")
    ax = plt.gca()
    ax.set_facecolor("#05060a")
    ax.imshow(dens, cmap="magma", origin="lower", alpha=0.92)
    ax.set_xticks([])
    ax.set_yticks([])

    # Stars glow
    xs = xi.astype(float)
    ys = yi.astype(float)
    _glow_scatter(ax, xs, ys, s=6 + 22 * s01, alpha=0.7, color="#9fd0ff")

    # Optional graph edges overlay
    if G_opt is not None and "source_id" in df.columns:
        sid_to_xy = {str(sid): (float(X), float(Y)) for sid, X, Y in zip(df["source_id"].astype(str), xs, ys, strict=False)}
        # Keep only a thin subset of edges to avoid clutter
        try:
            edges = list(G_opt.edges())
        except Exception:
            edges = []
        rng = np.random.default_rng(42)
        if len(edges) > 2500:
            idx = rng.choice(len(edges), size=2500, replace=False)
            edges = [edges[i] for i in idx.tolist()]
        for u, v in edges:
            su, sv = str(u), str(v)
            if su in sid_to_xy and sv in sid_to_xy:
                x0, y0 = sid_to_xy[su]
                x1, y1 = sid_to_xy[sv]
                ax.plot([x0, x1], [y0, y1], color="#2a3a66", alpha=0.10, linewidth=0.6)

    ax.set_title("Sky map anomalies", color="#eaeaea", pad=14)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def export_celestial_sphere(
    df: pd.DataFrame,
    out_html: Path,
    color_mode: str,
    phi_prefix: str,
    phi_weights: str,
    rgb_phis: str,
) -> None:
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    if "ra" not in df.columns or "dec" not in df.columns:
        out_html.write_text("Missing ra/dec in scored.csv", encoding="utf-8")
        return

    df2, cmap = _prepare_incoherence(df, color_mode, phi_prefix, phi_weights, rgb_phis)

    ra = np.deg2rad(pd.to_numeric(df2["ra"], errors="coerce").fillna(0.0).to_numpy(float))
    dec = np.deg2rad(pd.to_numeric(df2["dec"], errors="coerce").fillna(0.0).to_numpy(float))

    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)

    score = df2["incoherence_score"].to_numpy(float)
    size = 3 + 10 * score

    hover_cols = [c for c in ["source_id", "anomaly_score_hi", "incoherence_score", "incoherence_dominant", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"] if c in df2.columns]
    hover = df2[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

    mode = (color_mode or "score").strip().lower()

    fig = go.Figure()

    if mode == "dominant_phi" and ("incoherence_dominant" in df2.columns) and df2["incoherence_dominant"].astype(str).str.len().gt(0).any():
        dom = df2["incoherence_dominant"].astype(str).to_numpy()
        uniq = sorted(set(dom.tolist()))
        for name in uniq:
            mask = dom == name
            col = cmap.get(name, "#9fd0ff")
            fig.add_trace(go.Scatter3d(
                x=x[mask], y=y[mask], z=z[mask],
                mode="markers",
                name=name,
                marker=dict(size=size[mask], color=col, opacity=0.85),
                text=hover[mask] if hover is not None else None,
                hoverinfo="text",
            ))
        title = "Celestial sphere 3D, dominant incoherence"
    elif mode == "rgb_phi" and ("incoherence_rgb" in df2.columns) and df2["incoherence_rgb"].astype(str).str.len().gt(0).any():
        colors = df2["incoherence_rgb"].astype(str).to_list()
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=size, color=colors, opacity=0.85),
            text=hover,
            hoverinfo="text",
            name="rgb_phi",
        ))
        title = "Celestial sphere 3D, RGB incoherence mix"
    else:
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=size, color=score, colorscale="Viridis", opacity=0.85, colorbar=dict(title="incoherence")),
            text=hover,
            hoverinfo="text",
            name="score",
        ))
        title = "Celestial sphere 3D, incoherence score"

    fig.update_layout(
        title=title,
        height=900,
        dragmode="orbit",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=48),
        legend=dict(itemsizing="constant"),
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def export_network_explorer(
    df: pd.DataFrame,
    G_opt,
    out_html: Path,
    color_mode: str,
    phi_prefix: str,
    phi_weights: str,
    rgb_phis: str,
    max_nodes: int = 1200,
) -> None:
    if not _HAS_PYVIS:
        out_html.write_text("PyVis not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    if G_opt is None:
        out_html.write_text("No graph provided (graph_full or union.graphml).", encoding="utf-8")
        return

    df2, cmap = _prepare_incoherence(df, color_mode, phi_prefix, phi_weights, rgb_phis)
    G = G_opt
    nodes_keep, edges_keep = sample_subgraph(G, df2, max_nodes=int(max_nodes))

    d = df2.copy()
    if "source_id" not in d.columns:
        d["source_id"] = d.index.astype(str)
    d = d.set_index("source_id", drop=False)

    net = Network(height="820px", width="100%", bgcolor="#05060a", font_color="#e8e8e8", directed=False)
    net.force_atlas_2based(gravity=-30, central_gravity=0.01, spring_length=110, spring_strength=0.08, damping=0.4)

    mode = (color_mode or "score").strip().lower()

    for sid in nodes_keep:
        row = d.loc[sid] if sid in d.index else None
        sc = float(row["incoherence_score"]) if row is not None and "incoherence_score" in row else 0.2
        size = 10 + 30 * sc
        label = sid if sc > 0.92 else ""
        title_parts = []
        if row is not None:
            cols = ["source_id", "anomaly_score_hi", "incoherence_score", "incoherence_dominant", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe", "degree", "kcore", "betweenness"]
            for c in cols:
                if c in row.index:
                    title_parts.append(f"{c}: {row[c]}")
        title = "<br>".join(title_parts) if title_parts else sid

        if mode == "dominant_phi" and row is not None:
            dom = str(row.get("incoherence_dominant", ""))
            color = cmap.get(dom, "#9fd0ff")
        elif mode == "rgb_phi" and row is not None:
            color = str(row.get("incoherence_rgb", "rgb(160,160,255)")) or "rgb(160,160,255)"
        else:
            r = int(40 + 215 * sc)
            g = int(80 + 150 * sc)
            b = int(220 - 160 * sc)
            color = f"rgb({r},{g},{b})"

        net.add_node(sid, label=label, title=title, value=size, color=color)

    for u, v in edges_keep:
        net.add_edge(u, v, value=1)

    net.save_graph(str(out_html))


def load_lime_matrix(explain_path: Optional[Path]) -> Tuple[np.ndarray, List[str], List[str]]:
    if explain_path is None or not explain_path.exists():
        return np.zeros((0, 0), dtype=float), [], []
    rows = []
    ids = []
    feats_set = set()
    with explain_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            sid = str(obj.get("source_id", ""))
            lime = obj.get("lime") or {}
            weights = lime.get("weights") or []
            if not sid or not isinstance(weights, list):
                continue
            d = {}
            for w in weights:
                if not isinstance(w, dict):
                    continue
                feat = w.get("feature")
                val = w.get("weight")
                if feat is None:
                    continue
                try:
                    val_f = float(val)
                except Exception:
                    continue
                d[str(feat)] = val_f
                feats_set.add(str(feat))
            if d:
                rows.append(d)
                ids.append(sid)

    feats = sorted(feats_set)
    if not feats or not rows:
        return np.zeros((0, 0), dtype=float), [], []

    M = np.zeros((len(rows), len(feats)), dtype=float)
    for i, d in enumerate(rows):
        for j, feat in enumerate(feats):
            if feat in d:
                M[i, j] = float(d[feat])
    return M, ids, feats


def plot_explainability_heatmap(df: pd.DataFrame, explain_path: Optional[Path], out_png: Path, top_n: int = 40) -> None:
    M, ids, feats = load_lime_matrix(explain_path)
    if M.size == 0:
        # fallback: robust z scores on numeric columns
        ignore = {"source_id"}
        num_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
        cols = [c for c in num_cols if c not in ["anomaly_score", "anomaly_score_hi", "anomaly_score_norm"]][: min(20, len(num_cols))]
        if len(cols) < 2:
            fig = plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "No explanations and not enough numeric columns", ha="center", va="center")
            plt.axis("off")
            fig.savefig(out_png, dpi=260, bbox_inches="tight")
            plt.close(fig)
            return

        X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        Z = np.column_stack([robust_z(X[:, j]) for j in range(X.shape[1])])
        # Select top anomalies
        idx = np.argsort(-df["incoherence_score"].to_numpy(float))[: min(top_n, len(df))]
        Zs = Z[idx, :]
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(Zs, aspect="auto", cmap="coolwarm", vmin=-4, vmax=4)
        plt.yticks(range(len(idx)), df.iloc[idx]["source_id"].astype(str).tolist(), fontsize=6)
        plt.xticks(range(len(cols)), cols, rotation=75, fontsize=8)
        plt.title("Explainability heatmap, fallback robust z scores")
        plt.colorbar(label="robust z")
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        return

    # Use LIME matrix
    # Order rows by incoherence score if possible
    df_id = df.copy()
    if "source_id" not in df_id.columns:
        df_id["source_id"] = df_id.index.astype(str)
    df_id = df_id.set_index("source_id", drop=False)
    scores = []
    for sid in ids:
        if sid in df_id.index:
            scores.append(float(df_id.loc[sid].get("incoherence_score", 0.0)))
        else:
            scores.append(0.0)
    order = np.argsort(-np.asarray(scores))
    order = order[: min(top_n, len(order))]
    M2 = M[order, :]
    ids2 = [ids[i] for i in order.tolist()]

    # Keep top features by variance
    v = np.var(M2, axis=0)
    feat_order = np.argsort(-v)
    feat_order = feat_order[: min(25, len(feat_order))]
    M3 = M2[:, feat_order]
    feats3 = [feats[i] for i in feat_order.tolist()]

    fig = plt.figure(figsize=(12, 8))
    plt.imshow(M3, aspect="auto", cmap="coolwarm", vmin=-0.8, vmax=0.8)
    plt.yticks(range(len(ids2)), ids2, fontsize=6)
    plt.xticks(range(len(feats3)), feats3, rotation=75, fontsize=8)
    plt.title("Explainability heatmap, LIME weights")
    plt.colorbar(label="weight")
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def plot_feature_interaction_heatmap(df: pd.DataFrame, out_png: Path) -> None:
    ignore = {"source_id", "incoherence_dominant", "incoherence_rgb"}
    num_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe", "degree", "betweenness"] if c in num_cols]
    cols = preferred if len(preferred) >= 4 else num_cols[: min(8, len(num_cols))]
    if len(cols) < 2:
        fig = plt.figure(figsize=(12, 6))
        plt.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=260, bbox_inches="tight")
        plt.close(fig)
        return

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    C = np.corrcoef(X.T)
    fig = plt.figure(figsize=(10, 8))
    plt.imshow(C, cmap="viridis", vmin=-1, vmax=1)
    plt.xticks(range(len(cols)), cols, rotation=70, fontsize=8)
    plt.yticks(range(len(cols)), cols, fontsize=8)
    plt.title("Feature interaction heatmap, correlation")
    plt.colorbar(label="corr")
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def export_proper_motion_trails(df: pd.DataFrame, out_gif: Path, top_k: int = 30, frames: int = 24) -> None:
    if not _HAS_IMAGEIO:
        out_gif.write_bytes(b"imageio not installed")
        return
    if "pmra" not in df.columns or "pmdec" not in df.columns:
        out_gif.write_bytes(b"Missing pmra or pmdec")
        return

    d = df.copy()
    d = d.sort_values("incoherence_score", ascending=False).head(top_k)
    pmra = pd.to_numeric(d["pmra"], errors="coerce").fillna(0.0).to_numpy(float)
    pmdec = pd.to_numeric(d["pmdec"], errors="coerce").fillna(0.0).to_numpy(float)
    s = d["incoherence_score"].to_numpy(float)

    # Simulate trails from origin
    imgs = []
    t = np.linspace(0.0, 1.0, frames)
    for tt in t:
        fig = plt.figure(figsize=(6, 6), facecolor="#05060a")
        ax = plt.gca()
        ax.set_facecolor("#05060a")
        ax.scatter(pmra * tt, pmdec * tt, s=30 + 80 * s, c=s, alpha=0.85)
        ax.axhline(0, color="#334", alpha=0.3)
        ax.axvline(0, color="#334", alpha=0.3)
        ax.set_xlabel("pmra * t")
        ax.set_ylabel("pmdec * t")
        ax.set_title("Proper motion trails")
        ax.grid(alpha=0.15)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(img)
        plt.close(fig)

    imageio.mimsave(str(out_gif), imgs, duration=0.08)


def export_feature_biocubes(df: pd.DataFrame, out_html: Path) -> None:
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    ignore = {"source_id", "incoherence_dominant", "incoherence_rgb"}
    num_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe", "degree", "betweenness"] if c in num_cols]
    cols = preferred if len(preferred) >= 3 else num_cols[:3]
    if len(cols) < 3:
        out_html.write_text("Not enough numeric columns for biocubes", encoding="utf-8")
        return

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    score = df["incoherence_score"].to_numpy(float)
    fig = go.Figure(data=[go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=X[:, 2],
        mode="markers",
        marker=dict(size=4 + 10 * score, color=score, colorscale="Viridis", opacity=0.86, colorbar=dict(title="incoherence")),
        text=df.get("source_id", pd.Series(range(len(df)))).astype(str),
        hoverinfo="text",
    )])
    fig.update_layout(
        title=f"BioCubes 3D feature space: {cols[0]}, {cols[1]}, {cols[2]}",
        height=900,
        margin=dict(l=0, r=0, b=0, t=48),
        scene=dict(xaxis=dict(title=cols[0]), yaxis=dict(title=cols[1]), zaxis=dict(title=cols[2]), aspectmode="cube"),
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def export_umap(
    df: pd.DataFrame,
    out_png: Path,
    out_html: Path,
    color_mode: str,
    phi_prefix: str,
    phi_weights: str,
    rgb_phis: str,
) -> None:
    df2, cmap = _prepare_incoherence(df, color_mode, phi_prefix, phi_weights, rgb_phis)

    ignore = {"source_id", "incoherence_dominant", "incoherence_rgb"}
    num_cols = [c for c in df2.columns if c not in ignore and pd.api.types.is_numeric_dtype(df2[c])]
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

    X = df2[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.18, random_state=42)
        emb = reducer.fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        emb = U[:, :2] * S[:2]

    mode = (color_mode or "score").strip().lower()
    score = df2["incoherence_score"].to_numpy(float)

    fig = plt.figure(figsize=(10, 8))
    if mode == "dominant_phi" and df2["incoherence_dominant"].astype(str).str.len().gt(0).any():
        dom = df2["incoherence_dominant"].astype(str).to_numpy()
        for name, col in cmap.items():
            mask = dom == name
            plt.scatter(emb[mask, 0], emb[mask, 1], s=18, alpha=0.78, c=col, label=name)
        plt.legend(loc="best", fontsize=8, frameon=False)
    elif mode == "rgb_phi" and df2["incoherence_rgb"].astype(str).str.len().gt(0).any():
        colors = df2["incoherence_rgb"].astype(str).to_list()
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=colors)
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=score)
        plt.colorbar(label="incoherence_score")

    plt.title("Cosmic cloud embedding (UMAP if available)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    if _HAS_PLOTLY:
        hover_cols = [c for c in ["source_id", "anomaly_score_hi", "incoherence_score", "incoherence_dominant", "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "ruwe"] if c in df2.columns]
        hover = df2[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

        fig2 = go.Figure()
        if mode == "dominant_phi" and df2["incoherence_dominant"].astype(str).str.len().gt(0).any():
            dom = df2["incoherence_dominant"].astype(str).to_numpy()
            uniq = sorted(set(dom.tolist()))
            for name in uniq:
                mask = dom == name
                col = cmap.get(name, "#9fd0ff")
                fig2.add_trace(go.Scattergl(
                    x=emb[mask, 0], y=emb[mask, 1],
                    mode="markers",
                    name=name,
                    marker=dict(size=6 + 9 * score[mask], color=col, opacity=0.85),
                    text=hover[mask] if hover is not None else None,
                    hoverinfo="text",
                ))
        elif mode == "rgb_phi" and df2["incoherence_rgb"].astype(str).str.len().gt(0).any():
            colors = df2["incoherence_rgb"].astype(str).to_list()
            fig2.add_trace(go.Scattergl(
                x=emb[:, 0], y=emb[:, 1],
                mode="markers",
                marker=dict(size=6 + 9 * score, color=colors, opacity=0.85),
                text=hover,
                hoverinfo="text",
                name="rgb_phi",
            ))
        else:
            fig2.add_trace(go.Scattergl(
                x=emb[:, 0], y=emb[:, 1],
                mode="markers",
                marker=dict(size=6 + 9 * score, color=score, colorscale="Viridis", opacity=0.85, colorbar=dict(title="incoherence")),
                text=hover,
                hoverinfo="text",
                name="score",
            ))

        fig2.update_layout(title="UMAP cosmic cloud (interactive)", height=860, margin=dict(l=0, r=0, b=0, t=48))
        plotly_plot(fig2, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")
    else:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")


def plot_hr_cmd_outliers(
    df: pd.DataFrame,
    out_png: Path,
    out_html: Optional[Path],
    color_mode: str,
    phi_prefix: str,
    phi_weights: str,
    rgb_phis: str,
) -> None:
    """HR diagram style plot: absolute G magnitude versus BP RP color, highlight incoherent points."""
    df2, cmap = _prepare_incoherence(df, color_mode, phi_prefix, phi_weights, rgb_phis)

    required = ["phot_g_mean_mag", "bp_rp", "parallax"]
    if not all(c in df2.columns for c in required):
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Missing columns for HR CMD: phot_g_mean_mag, bp_rp, parallax", ha="center", va="center")
        plt.axis("off")
        fig.savefig(out_png, dpi=260, bbox_inches="tight")
        plt.close(fig)
        if out_html is not None:
            out_html.write_text("Missing columns for HR CMD", encoding="utf-8")
        return

    m = pd.to_numeric(df2["phot_g_mean_mag"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
    c = pd.to_numeric(df2["bp_rp"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)
    p = pd.to_numeric(df2["parallax"], errors="coerce").replace([np.inf, -np.inf], np.nan).to_numpy(float)

    # absolute magnitude from parallax in mas: M = m - 10 + 5*log10(parallax_mas)
    p2 = np.where(p > 0, p, np.nan)
    Mg = m - 10.0 + 5.0 * np.log10(p2)
    score = df2["incoherence_score"].to_numpy(float)

    mask_finite = np.isfinite(Mg) & np.isfinite(c)
    Mg = Mg[mask_finite]
    cc = c[mask_finite]
    ss = score[mask_finite]

    mode = (color_mode or "score").strip().lower()

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    ax.set_facecolor("#05060a")
    fig.patch.set_facecolor("#05060a")

    # background points
    ax.scatter(cc, Mg, s=6, alpha=0.20, c="#a0a6b8", linewidths=0)

    # highlight top incoherence
    if len(ss) > 0:
        topk = min(600, len(ss))
        idx = np.argsort(-ss)[:topk]
        if mode == "dominant_phi" and ("incoherence_dominant" in df2.columns) and df2["incoherence_dominant"].astype(str).str.len().gt(0).any():
            dom_all = df2.loc[mask_finite, "incoherence_dominant"].astype(str).to_numpy()
            dom = dom_all[idx]
            for name, col in cmap.items():
                m2 = dom == name
                if np.any(m2):
                    ax.scatter(cc[idx][m2], Mg[idx][m2], s=14, alpha=0.78, c=col, label=name, linewidths=0)
            ax.legend(loc="best", fontsize=8, frameon=False)
        elif mode == "rgb_phi" and ("incoherence_rgb" in df2.columns) and df2["incoherence_rgb"].astype(str).str.len().gt(0).any():
            colors_all = df2.loc[mask_finite, "incoherence_rgb"].astype(str).to_list()
            colors = [colors_all[i] for i in idx.tolist()]
            ax.scatter(cc[idx], Mg[idx], s=14, alpha=0.78, c=colors, linewidths=0)
        else:
            sc = ax.scatter(cc[idx], Mg[idx], s=14, alpha=0.78, c=ss[idx], cmap="viridis", linewidths=0)
            plt.colorbar(sc, ax=ax, label="incoherence_score")

    ax.set_title("HR CMD outliers (absolute G vs BP-RP)", color="#eaeaea", pad=12)
    ax.set_xlabel("BP-RP", color="#eaeaea")
    ax.set_ylabel("M_G (absolute)", color="#eaeaea")
    ax.tick_params(colors="#cfd4e6")
    ax.grid(alpha=0.12, color="#445")

    # brighter at top
    ax.invert_yaxis()

    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    if out_html is not None:
        if not _HAS_PLOTLY:
            out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
            return
        # Interactive version
        # reuse finite arrays
        hover_cols = [c for c in ["source_id", "anomaly_score_hi", "incoherence_score", "incoherence_dominant", "phot_g_mean_mag", "bp_rp", "parallax"] if c in df2.columns]
        hover = df2.loc[mask_finite, hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

        fig2 = go.Figure()
        if mode == "dominant_phi" and ("incoherence_dominant" in df2.columns) and df2["incoherence_dominant"].astype(str).str.len().gt(0).any():
            dom = df2.loc[mask_finite, "incoherence_dominant"].astype(str).to_numpy()
            uniq = sorted(set(dom.tolist()))
            for name in uniq:
                m3 = dom == name
                col = cmap.get(name, "#9fd0ff")
                fig2.add_trace(go.Scattergl(
                    x=cc[m3], y=Mg[m3],
                    mode="markers",
                    name=name,
                    marker=dict(size=5 + 9 * ss[m3], color=col, opacity=0.85),
                    text=hover[m3] if hover is not None else None,
                    hoverinfo="text",
                ))
        elif mode == "rgb_phi" and ("incoherence_rgb" in df2.columns) and df2["incoherence_rgb"].astype(str).str.len().gt(0).any():
            colors = df2.loc[mask_finite, "incoherence_rgb"].astype(str).to_list()
            fig2.add_trace(go.Scattergl(
                x=cc, y=Mg,
                mode="markers",
                marker=dict(size=5 + 9 * ss, color=colors, opacity=0.85),
                text=hover,
                hoverinfo="text",
                name="rgb_phi",
            ))
        else:
            fig2.add_trace(go.Scattergl(
                x=cc, y=Mg,
                mode="markers",
                marker=dict(size=5 + 9 * ss, color=ss, colorscale="Viridis", opacity=0.85, colorbar=dict(title="incoherence")),
                text=hover,
                hoverinfo="text",
                name="score",
            ))

        fig2.update_layout(
            title="HR CMD outliers (interactive)",
            height=860,
            margin=dict(l=0, r=0, b=0, t=48),
            yaxis=dict(autorange="reversed", title="M_G (absolute)"),
            xaxis=dict(title="BP-RP"),
        )
        plotly_plot(fig2, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def export_dashboard(out_dir: Path) -> None:
    rel = lambda p: p.name
    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>AstroGraphAnomaly A to H Gallery</title>
<style>
  body {{ background:#07080c; color:#eaeaea; font-family: ui-sans-serif, system-ui; margin: 24px; }}
  a {{ color:#8fb6ff; }}
  .grid {{ display:grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  .card {{ background:#0d1018; border:1px solid #1b2233; border-radius:14px; padding:14px; }}
  img {{ max-width: 100%; border-radius: 10px; }}
  h1,h2,h3 {{ margin: 8px 0; }}
  .links a {{ margin-right: 14px; }}
</style>
</head>
<body>
<h1>AstroGraphAnomaly A to H Gallery</h1>
<div class=\"links\">
  <a href=\"{rel(out_dir/'02_celestial_sphere_3d.html')}\">B Celestial Sphere 3D</a>
  <a href=\"{rel(out_dir/'03_network_explorer.html')}\">C Network Explorer</a>
  <a href=\"{rel(out_dir/'06_hr_cmd_outliers.html')}\">HR CMD interactive</a>
  <a href=\"{rel(out_dir/'08_feature_biocubes.html')}\">G BioCubes</a>
  <a href=\"{rel(out_dir/'10_umap_cosmic_cloud.html')}\">H UMAP interactive</a>
</div>

<h2>Curated visuals</h2>
<div class=\"grid\">
  <div class=\"card\"><h3>A Sky map anomalies</h3><img src=\"{rel(out_dir/'01_hidden_constellations_sky.png')}\"></div>
  <div class=\"card\"><h3>H UMAP cosmic cloud</h3><img src=\"{rel(out_dir/'09_umap_cosmic_cloud.png')}\"></div>
  <div class=\"card\"><h3>HR CMD outliers</h3><img src=\"{rel(out_dir/'06_hr_cmd_outliers.png')}\"></div>
  <div class=\"card\"><h3>D Explainability heatmap</h3><img src=\"{rel(out_dir/'04_explainability_heatmap.png')}\"></div>
  <div class=\"card\"><h3>Feature interaction heatmap</h3><img src=\"{rel(out_dir/'05_feature_interaction_heatmap.png')}\"></div>
  <div class=\"card\"><h3>F Proper motion trails</h3><img src=\"{rel(out_dir/'07_proper_motion_trails.gif')}\"></div>
</div>

</body>
</html>
"""
    (out_dir / "06_explorer_dashboard.html").write_text(html, encoding="utf-8")


def parse_args():
    ap = argparse.ArgumentParser(description="Generate A to H visualization suite for a run directory.")
    ap.add_argument("--run-dir", required=True, help="Run directory, outputs will be written under <run-dir>/viz_a_to_h")
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--graph", default="", help="Path to graph graphml (optional). If empty, auto detect in run dir.")
    ap.add_argument("--explain", default="", help="Path to explanations.jsonl (optional)")

    ap.add_argument("--color-mode", default="score", choices=["score", "dominant_phi", "rgb_phi"], help="Color mode for sphere, network, UMAP, HR CMD.")
    ap.add_argument("--phi-prefix", default="phi_", help="Prefix for constraint columns. Default: phi_")
    ap.add_argument("--phi-weights", default="", help="Weights for constraints, example: graph=2.0,lof=1.0")
    ap.add_argument("--rgb-phis", default="", help="Three phi columns for rgb_phi, example: graph,lof,ocsvm")

    ap.add_argument("--network-max-nodes", type=int, default=1200, help="Max nodes in PyVis graph explorer")
    return ap.parse_args()


def main() -> int:
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

    plot_hidden_constellations(df, G, out_dir / "01_hidden_constellations_sky.png")
    export_celestial_sphere(df, out_dir / "02_celestial_sphere_3d.html", args.color_mode, args.phi_prefix, args.phi_weights, args.rgb_phis)
    export_network_explorer(df, G, out_dir / "03_network_explorer.html", args.color_mode, args.phi_prefix, args.phi_weights, args.rgb_phis, max_nodes=args.network_max_nodes)
    plot_explainability_heatmap(df, explain, out_dir / "04_explainability_heatmap.png", top_n=40)
    plot_feature_interaction_heatmap(df, out_dir / "05_feature_interaction_heatmap.png")
    plot_hr_cmd_outliers(df, out_dir / "06_hr_cmd_outliers.png", out_dir / "06_hr_cmd_outliers.html", args.color_mode, args.phi_prefix, args.phi_weights, args.rgb_phis)
    export_proper_motion_trails(df, out_dir / "07_proper_motion_trails.gif", top_k=30, frames=24)
    export_feature_biocubes(df, out_dir / "08_feature_biocubes.html")
    export_umap(df, out_dir / "09_umap_cosmic_cloud.png", out_dir / "10_umap_cosmic_cloud.html", args.color_mode, args.phi_prefix, args.phi_weights, args.rgb_phis)
    export_dashboard(out_dir)

    print("OK: wrote A to H gallery to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
