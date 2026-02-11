#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly Plotly 3D mini suite (optional)

This is a post-processing tool: it reads a pipeline run directory and produces
interactive Plotly HTML views.

Input run directory must contain:
- scored.csv
- graph_topk.graphml (preferred) or graph_full.graphml (fallback)

Output:
<run_dir>/viz_plotly_3d/
  - 01_star_cloud_xyz.html   (needs ra, dec, distance)
  - 02_celestial_sphere.html (needs ra, dec)
  - 03_graph_3d.html         (needs a graph file)

Install Plotly via: pip install -r requirements_viz.txt

Usage:
  python tools/plotly_3d_report.py --run-dir results/<run>
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

try:
    import plotly.graph_objects as go
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Plotly is required.\n"
        "Install with: pip install -r requirements_viz.txt\n"
        f"Import error: {e}"
    )


def radec_to_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def radec_distance_to_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray, distance: np.ndarray) -> np.ndarray:
    return radec_to_unit_xyz(ra_deg, dec_deg) * distance.reshape(-1, 1)


def infer_higher_more_anom(df: pd.DataFrame) -> bool:
    if "anomaly_score" not in df.columns:
        return True
    if "anomaly_label" not in df.columns:
        return True
    an = df[df["anomaly_label"] == -1]["anomaly_score"]
    no = df[df["anomaly_label"] != -1]["anomaly_score"]
    if len(an) == 0 or len(no) == 0:
        return True
    return float(an.median()) >= float(no.median())


def pick_rows_for_viz(df: pd.DataFrame, max_points: int, keep_top: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()

    rng = np.random.default_rng(seed)
    higher_more_anom = infer_higher_more_anom(df)

    if "anomaly_score" in df.columns:
        df_sorted = df.sort_values("anomaly_score", ascending=not higher_more_anom)
        df_top = df_sorted.head(int(min(keep_top, len(df_sorted))))
    else:
        df_top = df.head(int(min(keep_top, len(df))))

    remaining = df.drop(index=df_top.index)
    n_bg = int(max(0, max_points - len(df_top)))
    if n_bg <= 0 or len(remaining) == 0:
        return df_top

    idx = rng.choice(remaining.index.to_numpy(), size=min(n_bg, len(remaining)), replace=False)
    return pd.concat([df_top, remaining.loc[idx]], axis=0).copy()


def score_array(df: pd.DataFrame) -> np.ndarray:
    if "anomaly_score" not in df.columns:
        return np.zeros(len(df), dtype=float)
    s = pd.to_numeric(df["anomaly_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    fill = float(s.median()) if len(s.dropna()) else 0.0
    return s.fillna(fill).to_numpy(float)


def hover_text(df: pd.DataFrame, cols: Sequence[str]) -> List[str]:
    cols = [c for c in cols if c in df.columns]
    out: List[str] = []
    for _, r in df.iterrows():
        parts: List[str] = []
        if "source_id" in df.columns:
            parts.append(f"source_id={r['source_id']}")
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if not (math.isfinite(v)):
                    continue
                parts.append(f"{c}={v:.4g}")
            else:
                parts.append(f"{c}={v}")
        out.append("<br>".join(parts))
    return out


def write_plot(fig: "go.Figure", out_html: Path) -> None:
    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def plot_star_cloud_xyz(df_scored: pd.DataFrame, out_html: Path, max_points: int, keep_top: int, seed: int) -> bool:
    if not {"ra", "dec", "distance"}.issubset(df_scored.columns):
        return False

    df = df_scored.dropna(subset=["ra", "dec", "distance"]).copy()
    if len(df) == 0:
        return False

    df = pick_rows_for_viz(df, max_points=max_points, keep_top=keep_top, seed=seed)

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)
    dist = pd.to_numeric(df["distance"], errors="coerce").to_numpy(float)

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dist) & (dist > 0)
    df = df.iloc[np.where(m)[0]].copy()
    if len(df) == 0:
        return False

    xyz = radec_distance_to_xyz(ra[m], dec[m], dist[m])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=score_array(df), colorscale="Viridis"),
                text=hover_text(df, ["anomaly_score", "anomaly_label", "ra", "dec", "distance", "phot_g_mean_mag"]),
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title="Star cloud 3D (RA, Dec, distance → xyz)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    write_plot(fig, out_html)
    return True


def plot_celestial_sphere(df_scored: pd.DataFrame, out_html: Path, max_points: int, keep_top: int, seed: int) -> bool:
    if not {"ra", "dec"}.issubset(df_scored.columns):
        return False

    df = df_scored.dropna(subset=["ra", "dec"]).copy()
    if len(df) == 0:
        return False

    df = pick_rows_for_viz(df, max_points=max_points, keep_top=keep_top, seed=seed)

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)

    m = np.isfinite(ra) & np.isfinite(dec)
    df = df.iloc[np.where(m)[0]].copy()
    if len(df) == 0:
        return False

    xyz = radec_to_unit_xyz(ra[m], dec[m])

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=score_array(df), colorscale="Viridis"),
                text=hover_text(df, ["anomaly_score", "anomaly_label", "ra", "dec", "phot_g_mean_mag"]),
                hoverinfo="text",
            )
        ]
    )
    fig.update_layout(
        title="Celestial sphere 3D (unit vectors from RA/Dec)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    write_plot(fig, out_html)
    return True


def graph_positions_from_df(G: nx.Graph, df_scored: pd.DataFrame) -> Optional[Dict[str, Tuple[float, float, float]]]:
    if not {"source_id", "ra", "dec"}.issubset(df_scored.columns):
        return None

    df = df_scored[["source_id", "ra", "dec"] + (["distance"] if "distance" in df_scored.columns else [])].copy()
    df["source_id"] = df["source_id"].astype(str)

    nodes = {str(n) for n in G.nodes()}
    df = df[df["source_id"].isin(nodes)].dropna(subset=["ra", "dec"])
    if len(df) == 0:
        return None

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)

    if "distance" in df.columns:
        dist = pd.to_numeric(df["distance"], errors="coerce").to_numpy(float)
        dist = np.where(np.isfinite(dist) & (dist > 0), dist, 1.0)
        xyz = radec_distance_to_xyz(ra, dec, dist)
        d = np.linalg.norm(xyz, axis=1)
        scale = np.where(d > 0, np.log1p(d) / d, 1.0)
        xyz = xyz * scale.reshape(-1, 1)
    else:
        xyz = radec_to_unit_xyz(ra, dec)

    pos: Dict[str, Tuple[float, float, float]] = {}
    for sid, p in zip(df["source_id"].tolist(), xyz.tolist(), strict=False):
        if all(np.isfinite(p)):
            pos[str(sid)] = (float(p[0]), float(p[1]), float(p[2]))

    return pos if len(pos) >= 10 else None


def plot_graph_3d(G: nx.Graph, df_scored: pd.DataFrame, out_html: Path, seed: int) -> None:
    pos = graph_positions_from_df(G, df_scored)
    if pos is None:
        pos_raw = nx.spring_layout(G, dim=3, seed=seed)
        pos = {str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in pos_raw.items()}

    df = df_scored.copy()
    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype(str)

    score_map: Dict[str, float] = {}
    label_map: Dict[str, float] = {}
    if {"source_id", "anomaly_score"}.issubset(df.columns):
        t = df[["source_id", "anomaly_score"]].copy()
        t["anomaly_score"] = pd.to_numeric(t["anomaly_score"], errors="coerce")
        for sid, sc in zip(t["source_id"].tolist(), t["anomaly_score"].tolist(), strict=False):
            if sc is None or (isinstance(sc, float) and not math.isfinite(sc)):
                continue
            score_map[str(sid)] = float(sc)
    if {"source_id", "anomaly_label"}.issubset(df.columns):
        t = df[["source_id", "anomaly_label"]].copy()
        t["anomaly_label"] = pd.to_numeric(t["anomaly_label"], errors="coerce")
        for sid, lb in zip(t["source_id"].tolist(), t["anomaly_label"].tolist(), strict=False):
            if lb is None or (isinstance(lb, float) and not math.isfinite(lb)):
                continue
            label_map[str(sid)] = float(lb)

    nodes = [str(n) for n in G.nodes() if str(n) in pos]
    x = [pos[n][0] for n in nodes]
    y = [pos[n][1] for n in nodes]
    z = [pos[n][2] for n in nodes]
    scores = [score_map.get(n, 0.0) for n in nodes]
    labels = [label_map.get(n, 0.0) for n in nodes]
    sizes = [6 if lb == -1 else 3 for lb in labels]

    node_hover = []
    for n in nodes:
        parts = [f"source_id={n}"]
        if n in score_map:
            parts.append(f"anomaly_score={score_map[n]:.4g}")
        if n in label_map:
            parts.append(f"anomaly_label={int(label_map[n])}")
        node_hover.append("<br>".join(parts))

    xe: List[Optional[float]] = []
    ye: List[Optional[float]] = []
    ze: List[Optional[float]] = []
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su not in pos or sv not in pos:
            continue
        xe.extend([pos[su][0], pos[sv][0], None])
        ye.extend([pos[su][1], pos[sv][1], None])
        ze.extend([pos[su][2], pos[sv][2], None])

    edge_trace = go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode="lines",
        line=dict(width=2, color="rgba(160,160,160,0.45)"),
        hoverinfo="none",
    )
    node_trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode="markers",
        marker=dict(size=sizes, opacity=0.92, color=scores, colorscale="Viridis"),
        text=node_hover,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title="Graph 3D (top-k subgraph)",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="data"),
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False,
    )
    write_plot(fig, out_html)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate Plotly 3D HTML views from an AstroGraphAnomaly run directory.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--max-points", type=int, default=50000)
    ap.add_argument("--keep-top", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    scored_path = run_dir / "scored.csv"
    if not scored_path.exists():
        raise SystemExit(f"Missing: {scored_path}")

    df_scored = pd.read_csv(scored_path)

    out_dir = run_dir / "viz_plotly_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_xyz = plot_star_cloud_xyz(df_scored, out_dir / "01_star_cloud_xyz.html", args.max_points, args.keep_top, args.seed)
    ok_sphere = plot_celestial_sphere(df_scored, out_dir / "02_celestial_sphere.html", args.max_points, args.keep_top, args.seed)

    graph_path = run_dir / "graph_topk.graphml"
    if not graph_path.exists():
        graph_path = run_dir / "graph_full.graphml"

    if graph_path.exists():
        G = nx.read_graphml(graph_path)
        plot_graph_3d(G, df_scored, out_dir / "03_graph_3d.html", seed=args.seed)

    created = [p.name for p in out_dir.glob("*.html")]
    print(f"Created {len(created)} HTML files in: {out_dir}")
    if not ok_xyz:
        print("Note: 01_star_cloud_xyz.html not created (needs ra, dec, distance).")
    if not ok_sphere:
        print("Note: 02_celestial_sphere.html not created (needs ra, dec).")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
