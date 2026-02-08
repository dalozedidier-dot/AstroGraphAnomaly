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
    size = 3 + 10*score

    hover_cols = [c for c in ["source_id","anomaly_score_hi","phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe"] if c in df.columns]
    hover = df[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=size, color=score, opacity=0.85),
        text=hover
    )])
    fig.update_layout(
        title="Celestial Sphere — anomaly score",
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), aspectmode="data"),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


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

    X = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    C = X.corr(method="spearman").to_numpy(float)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(C, aspect="auto")
    ax.set_title("Feature interaction heatmap (Spearman)")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)


def export_proper_motion_trails(df: pd.DataFrame, out_gif: Path, top_k: int = 30, frames: int = 24) -> None:
    if not _HAS_IMAGEIO:
        out_gif.write_text("imageio not installed. Install requirements_viz.txt.", encoding="utf-8")
        return
    needed = {"ra","dec","pmra","pmdec"}
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
        tt = np.linspace(max(0.0, t-2.0), t, steps)
        for i in range(len(ra0)):
            ra_tr = ra0[i] + dra_deg_per_yr[i]*tt
            dec_tr = dec0[i] + ddec_deg_per_yr[i]*tt
            ax.plot(ra_tr, dec_tr, alpha=0.35 + 0.4*score[i], linewidth=1.0 + 1.2*score[i])

        ra_t = ra0 + dra_deg_per_yr*t
        dec_t = dec0 + ddec_deg_per_yr*t
        ax.scatter(ra_t, dec_t, s=40 + 140*score, alpha=0.85)

        ax.set_title("Proper motion trails (Top anomalies)")
        ax.set_xlabel("RA [deg]")
        ax.set_ylabel("Dec [deg]")
        ax.grid(alpha=0.15)
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        imgs.append(img)
        plt.close(fig)

    imageio.mimsave(out_gif, imgs, duration=0.11)


def export_feature_biocubes(df: pd.DataFrame, out_html: Path) -> None:
    if not _HAS_PLOTLY:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")
        return

    y = df["anomaly_label"].to_numpy(int) if "anomaly_label" in df.columns else np.ones(len(df), dtype=int)
    an = df[y == -1]
    no = df[y != -1]

    candidates = [c for c in ["phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","distance","ruwe","degree","kcore","betweenness"]
                  if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    feats = candidates[:8] if len(candidates) >= 5 else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:6]
    if len(feats) == 0:
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

            vx = [x0,x1,x1,x0,x0,x1,x1,x0]
            vy = [y0,y0,y1,y1,y0,y0,y1,y1]
            vz = [z0,z0,z0,z0,z1,z1,z1,z1]

            I = [0,0,0, 4,4,4, 0,0, 1,1, 2,2]
            J = [1,2,3, 5,6,7, 4,5, 2,6, 3,7]
            K = [2,3,1, 6,7,5, 5,6, 6,5, 7,6]

            meshes.append(go.Mesh3d(x=vx,y=vy,z=vz, i=I,j=J,k=K, opacity=0.25))
            meshes.append(go.Scatter3d(x=[i], y=[gi], z=[med], mode="markers", marker=dict(size=5), name=f"{f} {name}"))

    fig = go.Figure(data=meshes)
    fig.update_layout(
        title="Feature BioCubes (IQR boxes + median markers)",
        scene=dict(
            xaxis=dict(title="feature index", tickmode="array", tickvals=list(range(len(feats))), ticktext=feats),
            yaxis=dict(title="group", tickmode="array", tickvals=[0,1], ticktext=["Normal","Anomalous"]),
            zaxis=dict(title="value"),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=False
    )
    plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")


def export_umap(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    ignore = {"source_id"}
    num_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    preferred = [c for c in ["phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","distance","ruwe","degree","kcore","betweenness"] if c in num_cols]
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
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=score)
    plt.title("Cosmic cloud embedding (UMAP if available)")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(alpha=0.15)
    plt.colorbar(label="anomaly_score_norm")
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    if _HAS_PLOTLY:
        hover_cols = [c for c in ["source_id","anomaly_score_hi","phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe"] if c in df.columns]
        hover = df[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None
        fig2 = go.Figure(data=[go.Scattergl(
            x=emb[:, 0], y=emb[:, 1], mode="markers",
            marker=dict(size=6 + 9*score, color=score, opacity=0.85),
            text=hover
        )])
        fig2.update_layout(title="UMAP cosmic cloud (interactive)", margin=dict(l=0, r=0, b=0, t=40))
        plotly_plot(fig2, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")
    else:
        out_html.write_text("Plotly not installed. Install requirements_viz.txt.", encoding="utf-8")


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
</div>

<h2>Curated visuals</h2>
<div class="grid">
  <div class="card"><h3>A) Hidden Constellations</h3><img src="{rel(out_dir/'01_hidden_constellations_sky.png')}" /></div>
  <div class="card"><h3>H) UMAP Cosmic Cloud</h3><img src="{rel(out_dir/'09_umap_cosmic_cloud.png')}" /></div>
  <div class="card"><h3>D) Explainability Heatmap</h3><img src="{rel(out_dir/'04_explainability_heatmap.png')}" /></div>
  <div class="card"><h3>D) Feature Interaction</h3><img src="{rel(out_dir/'05_feature_interaction_heatmap.png')}" /></div>
  <div class="card"><h3>F) Proper Motion Trails</h3><img src="{rel(out_dir/'07_proper_motion_trails.gif')}" /></div>
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

    plot_hidden_constellations(df, G, out_dir / "01_hidden_constellations_sky.png")
    export_celestial_sphere(df, out_dir / "02_celestial_sphere_3d.html")
    export_network_explorer(df, G, out_dir / "03_network_explorer.html")
    plot_explainability_heatmap(df, explain, out_dir / "04_explainability_heatmap.png", top_n=40)
    plot_feature_interaction_heatmap(df, out_dir / "05_feature_interaction_heatmap.png")
    export_proper_motion_trails(df, out_dir / "07_proper_motion_trails.gif", top_k=30, frames=24)
    export_feature_biocubes(df, out_dir / "08_feature_biocubes.html")
    export_umap(df, out_dir / "09_umap_cosmic_cloud.png", out_dir / "10_umap_cosmic_cloud.html")
    export_dashboard(out_dir)

    print("OK: wrote A→H gallery to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
