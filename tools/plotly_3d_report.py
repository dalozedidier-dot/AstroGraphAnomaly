#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plotly_3d_report.py

Generate optional Plotly 3D HTML views from an AstroGraphAnomaly run folder.

Expected in --run-dir:
- scored.csv
- graph_full.graphml (optional but recommended)
- graph_topk.graphml (optional)

Writes:
<run_dir>/viz_plotly_3d/
  - 01_star_cloud_xyz.html
  - 02_celestial_sphere.html
  - 03_graph_topk_3d.html (if graph_topk.graphml exists)
  - 04_graph_full_3d.html (if graph_full.graphml exists)

Install:
  pip install "plotly>=5.20"

Usage:
  python tools/plotly_3d_report.py --run-dir results/<run>

Notes on "style":
- style=default is tuned for the most common usage: exploration + showcase readability
  It uses aspectmode=cube + larger height + orbit dragmode.
- style=scientific preserves data proportions via aspectmode=data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

import plotly.graph_objects as go


def radec_to_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x, y, z])


def robust_01(x: np.ndarray) -> np.ndarray:
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


def score_hi(df: pd.DataFrame) -> np.ndarray:
    if "anomaly_score" not in df.columns:
        return np.zeros(len(df), dtype=float)

    s = pd.to_numeric(df["anomaly_score"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.fillna(float(np.nanmedian(s.to_numpy(float))) if len(s.dropna()) else 0.0).to_numpy(float)

    if "anomaly_label" in df.columns:
        y = pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(1).to_numpy(int)
        an = s[y == -1]
        no = s[y != -1]
        if len(an) and len(no) and float(np.nanmean(an)) < float(np.nanmean(no)):
            s = -s

    return s


def hover_text(df: pd.DataFrame) -> List[str]:
    cols = [c for c in ["source_id", "anomaly_score", "anomaly_label", "ra", "dec", "distance"] if c in df.columns]
    out: List[str] = []
    for _, r in df.iterrows():
        parts: List[str] = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                if not math.isfinite(v):
                    continue
                if c in {"ra", "dec"}:
                    parts.append(f"{c}={v:.4f}")
                elif c == "distance":
                    parts.append(f"{c}={v:.1f}")
                else:
                    parts.append(f"{c}={v:.4f}")
            else:
                parts.append(f"{c}={v}")
        out.append("<br>".join(parts))
    return out


def write_html(fig: "go.Figure", out_html: Path) -> None:
    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs="cdn"), encoding="utf-8")


def apply_3d_layout(fig: "go.Figure", title: str, style: str, height: int) -> None:
    if style == "scientific":
        scene = dict(aspectmode="data")
        camera = dict(eye=dict(x=1.25, y=1.25, z=1.05))
    else:
        scene = dict(aspectmode="cube")
        camera = dict(eye=dict(x=1.45, y=1.45, z=1.15))

    fig.update_layout(
        title=title,
        height=height,
        dragmode="orbit",
        scene={**scene, "camera": camera},
        margin=dict(l=0, r=0, t=55, b=0),
        showlegend=False,
    )


def depth_scale_default(dist: np.ndarray) -> np.ndarray:
    """
    Default scaling tuned for "most used" perception: give depth without exploding ranges.

    sqrt(dist) tends to keep relief without flattening as much as log.
    Then normalize by median to keep a stable dynamic range across runs.
    """
    d = np.asarray(dist, dtype=float)
    d = np.where(np.isfinite(d) & (d > 0), d, np.nan)
    rr = np.sqrt(d)
    med = float(np.nanmedian(rr)) if np.isfinite(np.nanmedian(rr)) else 1.0
    if med <= 0 or not math.isfinite(med):
        med = 1.0
    rr = rr / med
    rr = np.where(np.isfinite(rr), rr, 1.0)
    p99 = float(np.nanpercentile(rr, 99)) if len(rr) else 1.0
    if math.isfinite(p99) and p99 > 0:
        rr = np.clip(rr, 0.0, p99)
    return rr


def plot_star_cloud(df: pd.DataFrame, out_html: Path, style: str, height: int) -> bool:
    if not {"ra", "dec", "distance"}.issubset(df.columns):
        return False

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)
    dist = pd.to_numeric(df["distance"], errors="coerce").to_numpy(float)

    m = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(dist) & (dist > 0)
    if not np.any(m):
        return False

    uv = radec_to_unit_xyz(ra[m], dec[m])
    rr = depth_scale_default(dist[m])
    xyz = uv * rr.reshape(-1, 1)

    dfm = df.iloc[np.where(m)[0]]
    s = score_hi(dfm)
    c = robust_01(s)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=c, colorscale="Viridis"),
                text=hover_text(dfm),
                hoverinfo="text",
            )
        ]
    )
    apply_3d_layout(fig, "Star cloud 3D (RA, Dec with depth scaling)", style=style, height=height)
    write_html(fig, out_html)
    return True


def plot_celestial_sphere(df: pd.DataFrame, out_html: Path, style: str, height: int) -> bool:
    if not {"ra", "dec"}.issubset(df.columns):
        return False

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)

    m = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(m):
        return False

    xyz = radec_to_unit_xyz(ra[m], dec[m])
    dfm = df.iloc[np.where(m)[0]]
    s = score_hi(dfm)
    c = robust_01(s)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                marker=dict(size=3, opacity=0.85, color=c, colorscale="Viridis"),
                text=hover_text(dfm),
                hoverinfo="text",
            )
        ]
    )
    apply_3d_layout(fig, "Celestial sphere 3D (unit vectors from RA/Dec)", style=style, height=height)
    write_html(fig, out_html)
    return True


def graph_positions_from_node_attrs(G: nx.Graph) -> dict[str, Tuple[float, float, float]]:
    pos: dict[str, Tuple[float, float, float]] = {}
    for n, d in G.nodes(data=True):
        try:
            ra = float(d.get("ra"))
            dec = float(d.get("dec"))
            dist = float(d.get("distance", 1.0))
            if not (math.isfinite(ra) and math.isfinite(dec) and math.isfinite(dist) and dist > 0):
                continue
            uv = radec_to_unit_xyz(np.array([ra]), np.array([dec]))[0]
            rr = depth_scale_default(np.array([dist]))[0]
            pos[str(n)] = (float(uv[0] * rr), float(uv[1] * rr), float(uv[2] * rr))
        except Exception:
            continue
    return pos


def plot_graph_3d(G: nx.Graph, df: pd.DataFrame, out_html: Path, title: str, style: str, height: int) -> None:
    pos = graph_positions_from_node_attrs(G)
    if len(pos) < 10:
        raw = nx.spring_layout(G, dim=3, seed=42)
        pos = {str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in raw.items()}

    df2 = df.copy()
    if "source_id" in df2.columns:
        df2["source_id"] = df2["source_id"].astype(str)

    score_map = {}
    label_map = {}

    if {"source_id", "anomaly_score"}.issubset(df2.columns):
        scores = pd.to_numeric(df2["anomaly_score"], errors="coerce").to_numpy(float)
        for sid, sc in zip(df2["source_id"].tolist(), scores.tolist(), strict=False):
            if isinstance(sc, float) and math.isfinite(sc):
                score_map[str(sid)] = float(sc)

    if {"source_id", "anomaly_label"}.issubset(df2.columns):
        labels = pd.to_numeric(df2["anomaly_label"], errors="coerce").fillna(1).to_numpy(int)
        for sid, lb in zip(df2["source_id"].tolist(), labels.tolist(), strict=False):
            try:
                label_map[str(sid)] = int(lb)
            except Exception:
                label_map[str(sid)] = 1

    nodes = [str(n) for n in G.nodes() if str(n) in pos]
    scores = np.array([score_map.get(n, 0.0) for n in nodes], dtype=float)
    colors = robust_01(scores)
    sizes = [6 if label_map.get(n, 1) == -1 else 3 for n in nodes]
    htxt = [f"source_id={n}<br>score={score_map.get(n,0.0):.4f}<br>label={label_map.get(n,1)}" for n in nodes]

    x = [pos[n][0] for n in nodes]
    y = [pos[n][1] for n in nodes]
    z = [pos[n][2] for n in nodes]

    xe: List[Optional[float]] = []
    ye: List[Optional[float]] = []
    ze: List[Optional[float]] = []
    for u, v in G.edges():
        su, sv = str(u), str(v)
        if su in pos and sv in pos:
            xe.extend([pos[su][0], pos[sv][0], None])
            ye.extend([pos[su][1], pos[sv][1], None])
            ze.extend([pos[su][2], pos[sv][2], None])

    edge_trace = go.Scatter3d(
        x=xe,
        y=ye,
        z=ze,
        mode="lines",
        line=dict(width=1, color="rgba(160,160,160,0.18)"),
        hoverinfo="none",
    )
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(size=sizes, opacity=0.90, color=colors, colorscale="Viridis"),
        text=htxt,
        hoverinfo="text",
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    apply_3d_layout(fig, title, style=style, height=height)
    write_html(fig, out_html)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--style", choices=["default", "scientific"], default="default")
    ap.add_argument("--height", type=int, default=900)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    scored = run_dir / "scored.csv"
    if not scored.exists():
        raise SystemExit(f"Missing: {scored}")

    df = pd.read_csv(scored)

    out_dir = run_dir / "viz_plotly_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_star_cloud(df, out_dir / "01_star_cloud_xyz.html", style=args.style, height=args.height)
    plot_celestial_sphere(df, out_dir / "02_celestial_sphere.html", style=args.style, height=args.height)

    gt = run_dir / "graph_topk.graphml"
    if gt.exists():
        Gt = nx.read_graphml(gt)
        plot_graph_3d(
            Gt,
            df,
            out_dir / "03_graph_topk_3d.html",
            title="Graph top-k 3D (sky-space embedding)",
            style=args.style,
            height=args.height,
        )

    gf = run_dir / "graph_full.graphml"
    if gf.exists():
        Gf = nx.read_graphml(gf)
        plot_graph_3d(
            Gf,
            df,
            out_dir / "04_graph_full_3d.html",
            title="Graph full 3D (sky-space embedding)",
            style=args.style,
            height=args.height,
        )

    print(f"[plotly_3d_report] wrote HTML into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
