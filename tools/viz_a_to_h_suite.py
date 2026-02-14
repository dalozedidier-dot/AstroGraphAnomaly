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




def export_proper_motion_trails(
    df: pd.DataFrame,
    out_gif: Path,
    *,
    top_k: int = 30,
    frames: int = 24,
    years: float = 6.0,
) -> None:
    """F) Proper Motion Trails (GIF).

    We extrapolate positions using pmra/pmdec (mas/yr) over a small synthetic time window.
    This is a visualization tool, not an astrometric propagator.
    """
    if not _HAS_IMAGEIO:
        _placeholder_html(out_gif.with_suffix(".html"), "Proper Motion Trails", "imageio not installed. Install requirements_viz.txt.")
        return

    cols = _resolve_ra_dec_cols(df)
    if cols is None or not {"pmra", "pmdec"}.issubset(set(df.columns)):
        # degrade gracefully: write a single frame placeholder
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Missing ra/dec or pmra/pmdec columns for trails", ha="center", va="center")
        plt.axis("off")
        out_gif.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_gif.with_suffix(".png")
        fig.savefig(tmp, dpi=240, bbox_inches="tight")
        plt.close(fig)
        imageio.mimsave(out_gif, [imageio.imread(tmp)], duration=0.25)
        try:
            tmp.unlink()
        except Exception:
            pass
        return

    ra_col, dec_col = cols
    d = df.copy()
    d["__score"] = d["viz_color_value"].to_numpy(float) if "viz_color_value" in d.columns else d["anomaly_score_norm"].to_numpy(float)
    d = d.sort_values("__score", ascending=False).head(int(top_k)).copy()

    ra0 = np.deg2rad(pd.to_numeric(d[ra_col], errors="coerce").fillna(0.0).to_numpy(float))
    dec0 = np.deg2rad(pd.to_numeric(d[dec_col], errors="coerce").fillna(0.0).to_numpy(float))
    pmra = pd.to_numeric(d["pmra"], errors="coerce").fillna(0.0).to_numpy(float)  # mas/yr (mu_alpha* cos(delta))
    pmdec = pd.to_numeric(d["pmdec"], errors="coerce").fillna(0.0).to_numpy(float)  # mas/yr

    # Convert mas to radians
    mas_to_rad = (1.0 / 3.6e6) * (math.pi / 180.0)
    out_gif.parent.mkdir(parents=True, exist_ok=True)

    imgs = []
    for i in range(int(frames)):
        t = (i / max(1, frames - 1)) * years
        # approximate: undo cos(dec) for alpha
        dra = (pmra * mas_to_rad * t) / np.clip(np.cos(dec0), 1e-6, None)
        ddec = (pmdec * mas_to_rad * t)
        ra = ra0 + dra
        dec = dec0 + ddec

        fig = plt.figure(figsize=(9.5, 9.5), facecolor="black")
        ax = plt.gca()
        ax.set_facecolor("black")
        ax.axis("off")

        # Plot trails from start to current
        for k in range(len(d)):
            x0 = np.cos(dec0[k]) * np.cos(ra0[k])
            y0 = np.cos(dec0[k]) * np.sin(ra0[k])
            z0 = np.sin(dec0[k])
            x1 = np.cos(dec[k]) * np.cos(ra[k])
            y1 = np.cos(dec[k]) * np.sin(ra[k])
            z1 = np.sin(dec[k])

            # 2D projection (x,y) for compact GIF
            col = d["viz_color"].iloc[k] if "viz_color" in d.columns else "#88aaff"
            ax.plot([x0, x1], [y0, y1], linewidth=1.2, alpha=0.75, color=col)
            ax.scatter([x1], [y1], s=18, alpha=0.95, color=col)

        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        tmp = out_gif.parent / f"__trail_{i:03d}.png"
        fig.savefig(tmp, dpi=220, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        imgs.append(imageio.imread(tmp))

    imageio.mimsave(out_gif, imgs, duration=0.16)
    # cleanup tmp frames
    for i in range(int(frames)):
        p = out_gif.parent / f"__trail_{i:03d}.png"
        try:
            p.unlink()
        except Exception:
            pass


def export_feature_biocubes(df: pd.DataFrame, out_html: Path) -> None:
    """G) Feature BioCubes (Plotly 3D).

    A compact 3D scatter over three informative features. Colors follow viz_color if present.
    """
    if not _HAS_PLOTLY:
        _placeholder_html(out_html, "Feature BioCubes 3D", "Plotly not installed. Install requirements_viz.txt.")
        return

    d = _downsample_df(df, max_points=20000)

    # Choose 3 axes
    candidates = [
        ("bp_rp", "parallax", "phot_g_mean_mag"),
        ("pmra", "pmdec", "phot_g_mean_mag"),
        ("parallax", "ruwe", "phot_g_mean_mag"),
    ]
    axis = None
    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    for a,b,c in candidates:
        if a in num_cols and b in num_cols and c in num_cols:
            axis=(a,b,c); break
    if axis is None:
        # fallback: first 3 numeric columns
        if len(num_cols) >= 3:
            axis=(num_cols[0], num_cols[1], num_cols[2])
        else:
            _placeholder_html(out_html, "Feature BioCubes 3D", "Not enough numeric columns to build a 3D cube.")
            return
    ax, ay, az = axis

    X = pd.to_numeric(d[ax], errors="coerce").fillna(0.0).to_numpy(float)
    Y = pd.to_numeric(d[ay], errors="coerce").fillna(0.0).to_numpy(float)
    Z = pd.to_numeric(d[az], errors="coerce").fillna(0.0).to_numpy(float)
    X, Y, Z = robust_z(X), robust_z(Y), robust_z(Z)

    size = d["viz_size"].to_numpy(float) if "viz_size" in d.columns else (3 + 10*d["anomaly_score_norm"].to_numpy(float))
    hover_cols = [c for c in ["source_id","viz_category","anomaly_score_hi",ax,ay,az,"community_id"] if c in d.columns]
    hover = d[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

    has_viz_color = "viz_color" in d.columns and d["viz_color"].notna().any()
    if has_viz_color:
        colors = d["viz_color"].fillna("#888888").astype(str).to_list()
        marker = dict(size=4 + 0.6*size, color=colors, opacity=0.88)
        title = f"Feature BioCubes 3D — {ax} / {ay} / {az} (incoherence colors)"
    else:
        val = d["viz_color_value"].to_numpy(float) if "viz_color_value" in d.columns else d["anomaly_score_norm"].to_numpy(float)
        marker = dict(size=4 + 0.6*size, color=val, colorscale="Viridis", opacity=0.88, colorbar=dict(title="Anomaly/Incoh"))
        title = f"Feature BioCubes 3D — {ax} / {ay} / {az}"

    fig = go.Figure(data=[go.Scatter3d(
        x=X, y=Y, z=Z,
        mode="markers",
        marker=marker,
        text=hover,
        hoverinfo="text" if hover is not None else "skip"
    )])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=ax,
            yaxis_title=ay,
            zaxis_title=az,
            bgcolor="#05060a",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    _write_plotly_html(fig, out_html, "Feature BioCubes 3D")


def export_umap(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    """H) UMAP cosmic cloud (PNG + Plotly HTML)."""
    d = _downsample_df(df, max_points=25000)

    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    preferred = [c for c in ["phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe","degree","kcore","betweenness"] if c in num_cols]
    cols = preferred if len(preferred) >= 6 else num_cols[:12]
    if len(cols) < 2:
        fig = plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Not enough numeric columns for embedding", ha="center", va="center")
        plt.axis("off")
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=320, bbox_inches="tight")
        plt.close(fig)
        _placeholder_html(out_html, "UMAP cosmic cloud", "Not enough numeric columns for embedding.")
        return

    X = d[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    X = np.array(X, dtype=float, copy=True)
    for j in range(X.shape[1]):
        X[:, j] = robust_z(X[:, j])

    if _HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.18, random_state=42)
        emb = reducer.fit_transform(X)
    else:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        emb = U[:, :2] * S[:2]

    val = d["viz_color_value"].to_numpy(float) if "viz_color_value" in d.columns else d["anomaly_score_norm"].to_numpy(float)
    size = d["viz_size"].to_numpy(float) if "viz_size" in d.columns else (3 + 10*val)

    has_viz_color = "viz_color" in d.columns and d["viz_color"].notna().any()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(10, 8))
    if has_viz_color:
        colors = d["viz_color"].fillna("#888888").astype(str).to_list()
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=colors)
        plt.title("UMAP cosmic cloud — incoherence colors")
    else:
        plt.scatter(emb[:, 0], emb[:, 1], s=18, alpha=0.78, c=val)
        plt.title("UMAP cosmic cloud")
        plt.colorbar(label="anomaly / incoherence")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    # Interactive HTML
    hover_cols = [c for c in ["source_id","viz_category","anomaly_score_hi","phot_g_mean_mag","bp_rp","parallax","pmra","pmdec","ruwe","community_id"] if c in d.columns]
    hover = d[hover_cols].astype(str).agg("<br>".join, axis=1) if hover_cols else None

    if not _HAS_PLOTLY:
        _placeholder_html(out_html, "UMAP cosmic cloud", "Plotly not installed. Install requirements_viz.txt.")
        return

    if has_viz_color:
        colors = d["viz_color"].fillna("#888888").astype(str).to_list()
        marker = dict(size=4 + 0.9*size, color=colors, opacity=0.88)
    else:
        marker = dict(size=4 + 0.9*size, color=val, colorscale="Viridis", opacity=0.88, colorbar=dict(title="Anomaly/Incoh"))

    fig2 = go.Figure(data=[go.Scattergl(
        x=emb[:, 0], y=emb[:, 1],
        mode="markers",
        marker=marker,
        text=hover,
        hoverinfo="text" if hover is not None else "skip"
    )])
    fig2.update_layout(title="UMAP cosmic cloud (interactive)", margin=dict(l=0, r=0, b=0, t=40))
    _write_plotly_html(fig2, out_html, "UMAP cosmic cloud")

def plot_hr_cmd_outliers(df: pd.DataFrame, out_png: Path, out_html: Path) -> None:
    """
    I) HR/CMD-style outliers (Hertzsprung–Russell / Color-Magnitude Diagram).
    Uses:
      - x: bp_rp (color)
      - y: absolute G magnitude (requires parallax + phot_g_mean_mag)
    """
    # Accept either:
    # - precomputed bp_rp
    # - or (phot_bp_mean_mag - phot_rp_mean_mag)
    if "bp_rp" in df.columns:
        bp_rp = pd.to_numeric(df["bp_rp"], errors="coerce").to_numpy(float)
    elif {"phot_bp_mean_mag", "phot_rp_mean_mag"}.issubset(set(df.columns)):
        bp = pd.to_numeric(df["phot_bp_mean_mag"], errors="coerce").to_numpy(float)
        rp = pd.to_numeric(df["phot_rp_mean_mag"], errors="coerce").to_numpy(float)
        bp_rp = bp - rp
    else:
        _placeholder_png(out_png, "HR/CMD outliers", "Missing required color columns (bp_rp or phot_bp_mean_mag+phot_rp_mean_mag).")
        _placeholder_html(out_html, "HR/CMD outliers", "Missing required color columns (bp_rp or phot_bp_mean_mag+phot_rp_mean_mag).")
        return

    if not {"phot_g_mean_mag", "parallax"}.issubset(set(df.columns)):
        _placeholder_png(out_png, "HR/CMD outliers", "Missing required columns (phot_g_mean_mag, parallax).")
        _placeholder_html(out_html, "HR/CMD outliers", "Missing required columns (phot_g_mean_mag, parallax).")
        return

    gmag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").to_numpy(float)
    plx = pd.to_numeric(df["parallax"], errors="coerce").to_numpy(float)

    # Absolute magnitude: M_G = G + 5*log10(parallax_mas) - 10
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_g = gmag + 5.0 * np.log10(np.clip(plx, 1e-6, None)) - 10.0

    mask = np.isfinite(bp_rp) & np.isfinite(abs_g)
    if mask.sum() < 10:
        _placeholder_png(out_png, "HR/CMD outliers", "Not enough finite points to plot.")
        _placeholder_html(out_html, "HR/CMD outliers", "Not enough finite points to plot.")
        return

    val = df["viz_color_value"].to_numpy(float) if "viz_color_value" in df.columns else df["anomaly_score_norm"].to_numpy(float)
    size = df["viz_size"].to_numpy(float) if "viz_size" in df.columns else (3 + 10*val)
    has_viz_color = "viz_color" in df.columns and df["viz_color"].notna().any()
    colors = df["viz_color"].fillna("#888888").astype(str).to_list() if has_viz_color else None

    # PNG
    fig = plt.figure(figsize=(10, 8))
    if has_viz_color and colors is not None:
        plt.scatter(bp_rp[mask], abs_g[mask], s=16, alpha=0.78, c=np.asarray(colors)[mask])
        plt.title("HR/CMD outliers — incoherence colors")
    else:
        plt.scatter(bp_rp[mask], abs_g[mask], s=16, alpha=0.78, c=val[mask])
        plt.colorbar(label="anomaly / incoherence")
        plt.title("HR/CMD outliers")
    plt.xlabel("BP-RP")
    plt.ylabel("M_G (absolute)")
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.15)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    plt.close(fig)

    # HTML


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
</div>

<h2>Curated visuals</h2>
<div class="grid">
  <div class="card"><h3>A) Hidden Constellations</h3><img src="{rel(out_dir/'01_hidden_constellations_sky.png')}" /></div>
  <div class="card"><h3>H) UMAP Cosmic Cloud</h3><img src="{rel(out_dir/'09_umap_cosmic_cloud.png')}" /></div>
  <div class="card"><h3>D) Explainability Heatmap</h3><img src="{rel(out_dir/'04_explainability_heatmap.png')}" /></div>
  <div class="card"><h3>D) Feature Interaction</h3><img src="{rel(out_dir/'05_feature_interaction_heatmap.png')}" /></div>
  <div class="card"><h3>F) Proper Motion Trails</h3><img src="{rel(out_dir/'07_proper_motion_trails.gif')}" /></div>
  <div class="card"><h3>I) HR/CMD outliers</h3><img src="{rel(out_dir/'11_hr_cmd_outliers.png')}" /></div>
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

    _safe_call(
        "A) Hidden Constellations",
        [(out_dir / "01_hidden_constellations_sky.png", "png")],
        lambda: plot_hidden_constellations(df, G, out_dir / "01_hidden_constellations_sky.png"),
    )
    _safe_call(
        "B) Celestial Sphere 3D",
        [(out_dir / "02_celestial_sphere_3d.html", "html")],
        lambda: export_celestial_sphere(df, out_dir / "02_celestial_sphere_3d.html"),
    )
    _safe_call(
        "C) Network Explorer",
        [(out_dir / "03_network_explorer.html", "html")],
        lambda: export_network_explorer(df, G, out_dir / "03_network_explorer.html"),
    )
    _safe_call(
        "D) Explainability Heatmap",
        [(out_dir / "04_explainability_heatmap.png", "png")],
        lambda: plot_explainability_heatmap(df, explain, out_dir / "04_explainability_heatmap.png", top_n=40),
    )
    _safe_call(
        "D) Feature Interaction Heatmap",
        [(out_dir / "05_feature_interaction_heatmap.png", "png")],
        lambda: plot_feature_interaction_heatmap(df, out_dir / "05_feature_interaction_heatmap.png"),
    )
    _safe_call(
        "F) Proper Motion Trails",
        [(out_dir / "07_proper_motion_trails.gif", "gif")],
        lambda: export_proper_motion_trails(df, out_dir / "07_proper_motion_trails.gif", top_k=30, frames=24),
    )
    _safe_call(
        "G) Feature BioCubes",
        [(out_dir / "08_feature_biocubes.html", "html")],
        lambda: export_feature_biocubes(df, out_dir / "08_feature_biocubes.html"),
    )
    _safe_call(
        "H) UMAP cosmic cloud",
        [(out_dir / "09_umap_cosmic_cloud.png", "png"), (out_dir / "10_umap_cosmic_cloud.html", "html")],
        lambda: export_umap(df, out_dir / "09_umap_cosmic_cloud.png", out_dir / "10_umap_cosmic_cloud.html"),
    )
    _safe_call(
        "I) HR/CMD outliers",
        [(out_dir / "11_hr_cmd_outliers.png", "png"), (out_dir / "12_hr_cmd_outliers.html", "html")],
        lambda: plot_hr_cmd_outliers(df, out_dir / "11_hr_cmd_outliers.png", out_dir / "12_hr_cmd_outliers.html"),
    )
    _safe_call(
        "E) Explorer Dashboard",
        [(out_dir / "06_explorer_dashboard.html", "html")],
        lambda: export_dashboard(out_dir),
    )

    print("OK: wrote A→H (+HR/CMD) gallery to:", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
