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
  - index.html
  - 01_star_cloud_xyz.html
  - 02_celestial_sphere.html
  - 03_graph_topk_3d.html
  - 04_graph_full_3d.html

With --animate (default):
  Play/Pause camera orbit on the two point-cloud views.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx

import plotly.graph_objects as go


_BG = "#05060a"
_ANOMALY_COLOR = "#ff3b6b"
_HALO = "rgba(255, 90, 140, 0.18)"
_AXIS = dict(
    showbackground=False,
    showgrid=False,
    zeroline=False,
    showticklabels=False,
    showspikes=False,
    title="",
)


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


def anomaly_mask(df: pd.DataFrame) -> np.ndarray:
    if "anomaly_label" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    y = pd.to_numeric(df["anomaly_label"], errors="coerce").fillna(1).to_numpy(int)
    return y == -1


def marker_sizes(c01: np.ndarray, df: Optional[pd.DataFrame] = None, base: float = 2.4, span: float = 4.2) -> np.ndarray:
    """Score weight, plus inverse G magnitude when phot_g_mean_mag is present."""
    size = base + span * np.clip(np.asarray(c01, dtype=float), 0.0, 1.0)
    if df is not None and "phot_g_mean_mag" in df.columns:
        mag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
        if mag.shape[0] == size.shape[0] and np.isfinite(mag).any():
            bright = 1.0 - robust_01(mag)
            size = size * (0.75 + 0.7 * bright)
    return size


def hover_text(df: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in ["source_id", "anomaly_score", "anomaly_label", "ra", "dec", "distance", "phot_g_mean_mag", "bp_rp", "ruwe"]
        if c in df.columns
    ]
    fmt = {
        "ra": "{:.4f}",
        "dec": "{:.4f}",
        "distance": "{:.1f}",
        "anomaly_score": "{:.4f}",
        "phot_g_mean_mag": "{:.2f}",
        "bp_rp": "{:.3f}",
        "ruwe": "{:.3f}",
    }
    pieces: Dict[str, np.ndarray] = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_float_dtype(s):
            f = fmt.get(c, "{:.4f}")
            pieces[c] = np.array([f"{c}={f.format(v)}" if math.isfinite(v) else "" for v in s.to_numpy(float)])
        else:
            pieces[c] = np.array([f"{c}={v}" for v in s.tolist()])
    rows = ["<br>".join(p for p in parts if p) for parts in zip(*[pieces[c] for c in cols])]
    return rows


def write_html(fig: "go.Figure", out_html: Path) -> None:
    out_html.write_text(fig.to_html(full_html=True, include_plotlyjs=True), encoding="utf-8")


def add_orbit_animation(fig: "go.Figure", *, frames: int = 72, radius: float = 1.85, z: float = 1.12) -> None:
    """Camera orbit around the scene. Frames only store camera pose (cheap)."""
    n = max(12, int(frames))
    fig.frames = [
        go.Frame(
            name=str(i),
            layout={
                "scene": {
                    "camera": {
                        "eye": {
                            "x": radius * math.cos(2 * math.pi * i / n),
                            "y": radius * math.sin(2 * math.pi * i / n),
                            "z": z,
                        }
                    }
                }
            },
        )
        for i in range(n)
    ]
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                bgcolor="rgba(12,14,22,0.75)",
                bordercolor="rgba(255,59,107,0.45)",
                font=dict(color="#e8e8ef"),
                buttons=[
                    dict(
                        label="▶ Play orbit",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=45, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=0),
                                mode="immediate",
                            ),
                        ],
                    ),
                    dict(
                        label="❚❚ Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")],
                    ),
                ],
            )
        ]
    )


def apply_3d_layout(fig: "go.Figure", title: str, style: str, height: int) -> None:
    if style == "scientific":
        aspect = dict(aspectmode="data")
        camera = dict(eye=dict(x=1.25, y=1.25, z=1.05))
    else:
        aspect = dict(aspectmode="cube")
        camera = dict(eye=dict(x=1.45, y=1.45, z=1.15))

    fig.update_layout(
        title=dict(text=title, font=dict(color="#e8e8ef", size=18)),
        height=height,
        template="plotly_dark",
        paper_bgcolor=_BG,
        dragmode="orbit",
        scene={
            **aspect,
            "camera": camera,
            "bgcolor": _BG,
            "xaxis": _AXIS,
            "yaxis": _AXIS,
            "zaxis": _AXIS,
        },
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e8e8ef")),
        showlegend=True,
    )


def _colorbar(title: str) -> dict:
    return dict(
        title=dict(text=title, side="right", font=dict(color="#e8e8ef")),
        thickness=12,
        len=0.6,
        tickfont=dict(color="#cfd0d8"),
        outlinewidth=0,
    )


def _field_trace(xyz: np.ndarray, c01: np.ndarray, text: List[str], colorscale: str, sizes: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name="sources",
        marker=dict(
            size=sizes,
            opacity=0.78,
            color=c01,
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            colorbar=_colorbar("Anomaly score (robust 0..1)"),
            showscale=True,
        ),
        text=text,
        hoverinfo="text",
    )


def _anomaly_halo(xyz: np.ndarray) -> go.Scatter3d:
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name="anomaly glow",
        marker=dict(size=16, color=_HALO, opacity=0.55, line=dict(width=0)),
        hoverinfo="skip",
        showlegend=False,
    )


def _anomaly_trace(xyz: np.ndarray, text: List[str]) -> go.Scatter3d:
    return go.Scatter3d(
        x=xyz[:, 0],
        y=xyz[:, 1],
        z=xyz[:, 2],
        mode="markers",
        name="anomalies",
        marker=dict(
            size=8,
            symbol="diamond",
            color=_ANOMALY_COLOR,
            opacity=0.95,
            line=dict(color="#ffffff", width=1),
        ),
        text=text,
        hoverinfo="text",
    )


def _reference_sphere(n_lines: int = 12, n_pts: int = 60) -> List[go.Scatter3d]:
    traces: List[go.Scatter3d] = []
    line = dict(color="rgba(120,140,180,0.18)", width=1)
    for dec in np.linspace(-75, 75, n_lines):
        ra = np.linspace(0, 360, n_pts)
        xyz = radec_to_unit_xyz(ra, np.full_like(ra, dec))
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line=line,
                hoverinfo="none",
                showlegend=False,
            )
        )
    for ra0 in np.linspace(0, 360, n_lines, endpoint=False):
        dec = np.linspace(-90, 90, n_pts)
        xyz = radec_to_unit_xyz(np.full_like(dec, ra0), dec)
        traces.append(
            go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="lines",
                line=line,
                hoverinfo="none",
                showlegend=False,
            )
        )
    return traces


def depth_scale_default(dist: np.ndarray) -> np.ndarray:
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


def plot_star_cloud(
    df: pd.DataFrame,
    out_html: Path,
    style: str,
    height: int,
    colorscale: str,
    animate: bool,
    orbit_frames: int,
) -> bool:
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
    c = robust_01(score_hi(dfm))
    text = hover_text(dfm)
    sizes = marker_sizes(c, dfm)

    data = [_field_trace(xyz, c, text, colorscale, sizes)]
    am = anomaly_mask(dfm)
    if np.any(am):
        data.append(_anomaly_halo(xyz[am]))
        data.append(_anomaly_trace(xyz[am], [t for t, k in zip(text, am) if k]))

    fig = go.Figure(data=data)
    apply_3d_layout(fig, "Star cloud 3D — RA/Dec with depth scaling", style=style, height=height)
    if animate:
        add_orbit_animation(fig, frames=orbit_frames)
    write_html(fig, out_html)
    return True


def plot_celestial_sphere(
    df: pd.DataFrame,
    out_html: Path,
    style: str,
    height: int,
    colorscale: str,
    animate: bool,
    orbit_frames: int,
) -> bool:
    if not {"ra", "dec"}.issubset(df.columns):
        return False

    ra = pd.to_numeric(df["ra"], errors="coerce").to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").to_numpy(float)

    m = np.isfinite(ra) & np.isfinite(dec)
    if not np.any(m):
        return False

    xyz = radec_to_unit_xyz(ra[m], dec[m])
    dfm = df.iloc[np.where(m)[0]]
    c = robust_01(score_hi(dfm))
    text = hover_text(dfm)
    sizes = marker_sizes(c, dfm)

    data: List[go.Scatter3d] = list(_reference_sphere())
    data.append(_field_trace(xyz, c, text, colorscale, sizes))
    am = anomaly_mask(dfm)
    if np.any(am):
        data.append(_anomaly_halo(xyz[am]))
        data.append(_anomaly_trace(xyz[am], [t for t, k in zip(text, am) if k]))

    fig = go.Figure(data=data)
    apply_3d_layout(fig, "Celestial sphere 3D — unit vectors from RA/Dec", style=style, height=height)
    if animate:
        add_orbit_animation(fig, frames=orbit_frames, radius=1.7, z=0.95)
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


def plot_graph_3d(G: nx.Graph, df: pd.DataFrame, out_html: Path, title: str, style: str, height: int, colorscale: str) -> None:
    pos = graph_positions_from_node_attrs(G)
    if len(pos) < 10:
        raw = nx.spring_layout(G, dim=3, seed=42)
        pos = {str(k): (float(v[0]), float(v[1]), float(v[2])) for k, v in raw.items()}

    df2 = df.copy()
    if "source_id" in df2.columns:
        df2["source_id"] = df2["source_id"].astype(str)

    score_map: Dict[str, float] = {}
    label_map: Dict[str, int] = {}

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
    htxt = [f"source_id={n}<br>score={score_map.get(n, 0.0):.4f}<br>label={label_map.get(n, 1)}" for n in nodes]

    x = np.array([pos[n][0] for n in nodes])
    y = np.array([pos[n][1] for n in nodes])
    z = np.array([pos[n][2] for n in nodes])

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
        line=dict(width=1, color="rgba(150,170,210,0.16)"),
        hoverinfo="none",
        name="edges",
        showlegend=False,
    )
    node_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        name="sources",
        marker=dict(
            size=marker_sizes(colors, base=3.0, span=3.5),
            opacity=0.9,
            color=colors,
            colorscale=colorscale,
            cmin=0.0,
            cmax=1.0,
            colorbar=_colorbar("Anomaly score (robust 0..1)"),
            showscale=True,
        ),
        text=htxt,
        hoverinfo="text",
    )

    data = [edge_trace, node_trace]
    am = np.array([label_map.get(n, 1) == -1 for n in nodes], dtype=bool)
    if np.any(am):
        xyz_a = np.column_stack([x[am], y[am], z[am]])
        data.append(_anomaly_halo(xyz_a))
        data.append(_anomaly_trace(xyz_a, [t for t, k in zip(htxt, am) if k]))

    fig = go.Figure(data=data)
    apply_3d_layout(fig, title, style=style, height=height)
    write_html(fig, out_html)


def write_index(out_dir: Path, entries: List[Tuple[str, str]], animated: bool) -> None:
    hint = (
        "Play orbit sur le nuage et la sphère · glisser pour orbiter · molette pour zoomer · anomalies = losanges roses + halo."
        if animated
        else "Glisser pour orbiter · molette pour zoomer · les anomalies sont en losanges roses."
    )
    cards = "".join(f'<a class="card" href="{href}"><span>{title}</span></a>' for title, href in entries)
    doc = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AstroGraphAnomaly — Vues 3D</title>
<style>
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
          background: {_BG}; color: #e8e8ef; margin: 0; padding: 32px; }}
  h1 {{ font-weight: 700; letter-spacing: .02em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 20px; }}
  .card {{ display: flex; align-items: flex-end; min-height: 120px; padding: 16px;
           border-radius: 14px; text-decoration: none; color: #fff; font-weight: 600;
           background: linear-gradient(135deg, #11162b, #1d2547);
           border: 1px solid rgba(120,140,200,.25); transition: transform .12s ease, border-color .12s; }}
  .card:hover {{ transform: translateY(-3px); border-color: {_ANOMALY_COLOR}; }}
</style></head>
<body>
  <h1>Vues 3D interactives</h1>
  <p style="color:#9aa0b5">{hint}</p>
  <div class="grid">{cards}</div>
</body></html>
"""
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--style", choices=["default", "scientific"], default="default")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--colorscale", default="Plasma", help="Plotly colorscale (e.g. Plasma, Viridis, Turbo, Inferno)")
    ap.add_argument("--animate", dest="animate", action="store_true", default=True)
    ap.add_argument("--no-animate", dest="animate", action="store_false")
    ap.add_argument("--orbit-frames", type=int, default=72)
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    scored = run_dir / "scored.csv"
    if not scored.exists():
        raise SystemExit(f"Missing: {scored}")

    df = pd.read_csv(scored)

    out_dir = run_dir / "viz_plotly_3d"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: List[Tuple[str, str]] = []
    if plot_star_cloud(
        df, out_dir / "01_star_cloud_xyz.html", args.style, args.height, args.colorscale, args.animate, args.orbit_frames
    ):
        entries.append(("Star cloud 3D", "01_star_cloud_xyz.html"))
    if plot_celestial_sphere(
        df, out_dir / "02_celestial_sphere.html", args.style, args.height, args.colorscale, args.animate, args.orbit_frames
    ):
        entries.append(("Celestial sphere 3D", "02_celestial_sphere.html"))

    gt = run_dir / "graph_topk.graphml"
    if gt.exists():
        plot_graph_3d(
            nx.read_graphml(gt),
            df,
            out_dir / "03_graph_topk_3d.html",
            "Graph top-k 3D — sky-space embedding",
            args.style,
            args.height,
            args.colorscale,
        )
        entries.append(("Graph top-k 3D", "03_graph_topk_3d.html"))

    gf = run_dir / "graph_full.graphml"
    if gf.exists():
        plot_graph_3d(
            nx.read_graphml(gf),
            df,
            out_dir / "04_graph_full_3d.html",
            "Graph full 3D — sky-space embedding",
            args.style,
            args.height,
            args.colorscale,
        )
        entries.append(("Graph full 3D", "04_graph_full_3d.html"))

    write_index(out_dir, entries, animated=bool(args.animate))
    print(f"[plotly_3d_report] wrote {len(entries)} views + index.html into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
