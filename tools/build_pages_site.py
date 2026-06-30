#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/build_pages_site.py

Assemble the GitHub Pages site from the project's own workflow outputs.

It runs a small, offline demo of AstroGraphAnomaly on the bundled sample
catalog (no network), generates the HTML report, the static plots and the
interactive Plotly 3D views, then lays everything out as a deployable static
site:

    <out>/
      index.html                 (the landing page, copied from repo root)
      assets/logos/*             (landing logos)
      demo/
        index.html               (themed gallery linking every artefact)
        report.html              (run report)
        summary.json
        plots/*.png
        viz_plotly_3d/*.html     (interactive 3D views)

The matching workflow (.github/workflows/deploy_pages_from_workflow_outputs.yml)
publishes <out> to GitHub Pages, so the "Démo en ligne" link on the landing page
resolves to a freshly generated demo on every deploy.

Usage:
  python tools/build_pages_site.py --out site
  python tools/build_pages_site.py --out site --run-dir results/existing_run
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for _p in (REPO_ROOT, SRC):  # repo root makes `tools` importable; src makes the package importable
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Shared palette with the landing page / 3D theme.
BG = "#05060a"
PINK = "#ff3b6b"
CYAN = "#00ffcc"
BLUE = "#00d4ff"

PLOT_TITLES = {
    "score_hist.png": "Distribution des scores",
    "ra_dec_score.png": "Carte spatiale (RA/Dec)",
    "cmd_bp_rp_vs_g.png": "Diagramme Couleur–Magnitude",
    "mag_vs_distance.png": "Magnitude vs distance",
    "mean_features_anom_vs_normal.png": "Features : anomal vs normal",
    "top_anomalies_scores.png": "Top anomalies",
    "pca_2d.png": "Projection PCA",
    "graph_communities_anomalies.png": "Graphe : communautés + anomalies",
}

VIEW_TITLES = {
    "01_star_cloud_xyz.html": "Star cloud 3D",
    "02_celestial_sphere.html": "Sphère céleste 3D",
    "03_graph_topk_3d.html": "Graphe top-k 3D",
    "04_graph_full_3d.html": "Graphe complet 3D",
}


def generate_demo_run(run_dir: Path) -> dict:
    """Run the offline demo pipeline (sample CSV, ensemble engine, plots)."""
    from astrographanomaly.pipeline import run_pipeline

    return run_pipeline(
        mode="csv",
        in_csv=str(REPO_ROOT / "data" / "sample_gaia_like.csv"),
        out_dir=str(run_dir),
        engine="ensemble",
        threshold_strategy="percentile",
        percentile=97.0,
        top_k=40,
        knn_k=8,
        features_mode="extended",
        explain_top=0,
        plots=True,
        seed=42,
    )


def generate_3d_views(run_dir: Path) -> None:
    """Generate the Plotly 3D views (best-effort: skipped if plotly missing)."""
    try:
        import plotly  # noqa: F401
    except Exception:
        print("[build_pages_site] plotly not installed -> skipping 3D views")
        return
    from tools.plotly_3d_report import main as plotly_main

    plotly_main(["--run-dir", str(run_dir), "--height", "820"])


def _card(href: str, title: str, sub: str, accent: str) -> str:
    return (
        f'<a class="card" style="--accent:{accent}" href="{html.escape(href)}">'
        f'<span class="ct">{html.escape(title)}</span>'
        f'<span class="cs">{html.escape(sub)}</span></a>'
    )


def _img_card(href: str, title: str) -> str:
    return (
        f'<a class="shot" href="{html.escape(href)}" target="_blank" rel="noopener">'
        f'<img src="{html.escape(href)}" alt="{html.escape(title)}" loading="lazy" />'
        f'<span>{html.escape(title)}</span></a>'
    )


def build_demo_index(demo_dir: Path, summary: dict) -> None:
    counts = summary.get("counts", {})
    stats = [
        (counts.get("n_rows", 0), "Sources"),
        (counts.get("n_anomalies", 0), "Anomalies"),
        (counts.get("n_edges", 0), "Arêtes k-NN"),
        (summary.get("engine", "—"), "Engine"),
    ]
    stat_html = "".join(
        f'<div class="stat"><div class="n">{html.escape(str(v))}</div>'
        f'<div class="l">{html.escape(l)}</div></div>'
        for v, l in stats
    )

    primary: List[str] = []
    if (demo_dir / "report.html").exists():
        primary.append(_card("report.html", "Rapport HTML", "Stats, top-anomalies & plots", PINK))
    views_dir = demo_dir / "viz_plotly_3d"
    for name, title in VIEW_TITLES.items():
        if (views_dir / name).exists():
            primary.append(_card(f"viz_plotly_3d/{name}", title, "Vue Plotly interactive", CYAN))
    cards_html = "".join(primary)

    shots: List[str] = []
    plots_dir = demo_dir / "plots"
    if plots_dir.is_dir():
        ordered = [n for n in PLOT_TITLES if (plots_dir / n).exists()]
        extras = sorted(p.name for p in plots_dir.glob("*.png") if p.name not in PLOT_TITLES)
        for n in ordered:
            shots.append(_img_card(f"plots/{n}", PLOT_TITLES[n]))
        for n in extras:
            shots.append(_img_card(f"plots/{n}", n))
    shots_html = (
        f'<h2>Visualisations statiques</h2><div class="shots">{"".join(shots)}</div>'
        if shots else ""
    )

    doc = f"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
<title>AstroGraphAnomaly — Démo</title>
<style>
  :root {{ --bg:{BG}; --pink:{PINK}; --cyan:{CYAN}; --blue:{BLUE}; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; color: #e8e8ff;
    background:
      radial-gradient(1100px 600px at 80% -10%, rgba(0,212,255,.10), transparent 60%),
      radial-gradient(900px 700px at 10% 110%, rgba(255,59,107,.10), transparent 60%),
      var(--bg);
    min-height: 100vh; padding: 40px clamp(16px,4vw,56px); }}
  a {{ color: inherit; text-decoration: none; }}
  .top {{ max-width: 1100px; margin: 0 auto 28px; }}
  .back {{ color: #9aa0c0; font-weight: 600; }}
  .back:hover {{ color: var(--cyan); }}
  h1 {{ font-size: clamp(1.8rem,5vw,2.6rem); margin: 14px 0 4px; letter-spacing: .5px; }}
  h1 span {{ color: var(--pink); }}
  .sub {{ color: #9aa0c0; }}
  .stats {{ display: flex; gap: 26px; flex-wrap: wrap; margin: 22px 0 8px; }}
  .stat .n {{ font-size: 1.7rem; font-weight: 800; }}
  .stat .l {{ font-size: .76rem; letter-spacing: 1px; text-transform: uppercase; color: #5a6090; }}
  .grid {{ max-width: 1100px; margin: 0 auto; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
  .card {{ display: flex; flex-direction: column; gap: 6px; min-height: 110px; padding: 18px;
    border-radius: 14px; background: rgba(18,20,42,.6); border: 1px solid rgba(120,140,200,.18);
    border-left: 3px solid var(--accent); transition: transform .14s, border-color .2s; }}
  .card:hover {{ transform: translateY(-4px); border-color: var(--accent); }}
  .ct {{ font-weight: 700; font-size: 1.1rem; }}
  .cs {{ color: #9aa0c0; font-size: .9rem; }}
  h2 {{ max-width: 1100px; margin: 40px auto 16px; font-size: 1.2rem; color: #cfd2e6; }}
  .shots {{ max-width: 1100px; margin: 0 auto; display: grid;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 16px; }}
  .shot {{ display: block; border-radius: 12px; overflow: hidden; background: #0c0d1d;
    border: 1px solid rgba(120,140,200,.18); }}
  .shot img {{ width: 100%; display: block; }}
  .shot span {{ display: block; padding: 10px 12px; font-size: .88rem; color: #cfd2e6; }}
  footer {{ max-width: 1100px; margin: 48px auto 0; color: #5a6090; font-size: .85rem;
    border-top: 1px solid rgba(120,140,200,.18); padding-top: 18px; }}
</style></head>
<body>
  <div class="top">
    <a class="back" href="../">← Toolkit</a>
    <h1>Astro<span>Graph</span>Anomaly — démo live</h1>
    <p class="sub">Run hors-ligne sur le catalogue d'exemple (engine ensemble), généré à chaque déploiement Pages.</p>
    <div class="stats">{stat_html}</div>
  </div>
  <div class="grid">{cards_html}</div>
  {shots_html}
  <footer>Généré par <code>tools/build_pages_site.py</code> · données : <code>data/sample_gaia_like.csv</code></footer>
</body></html>
"""
    (demo_dir / "index.html").write_text(doc, encoding="utf-8")


def assemble_site(out: Path, run_dir: Path, summary: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # Landing page + its assets.
    shutil.copy2(REPO_ROOT / "index.html", out / "index.html")
    assets_src = REPO_ROOT / "assets"
    if assets_src.is_dir():
        shutil.copytree(assets_src, out / "assets", dirs_exist_ok=True)

    # Demo payload.
    demo = out / "demo"
    if demo.exists():
        shutil.rmtree(demo)
    demo.mkdir(parents=True)

    for name in ("report.html", "summary.json"):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, demo / name)
    for sub in ("plots", "viz_plotly_3d"):
        src = run_dir / sub
        if src.is_dir():
            shutil.copytree(src, demo / sub, dirs_exist_ok=True)

    build_demo_index(demo, summary)
    # GitHub Pages: skip Jekyll processing so files with leading underscores survive.
    (out / ".nojekyll").write_text("", encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="site", help="Output site directory")
    ap.add_argument("--run-dir", default=None, help="Reuse an existing run dir instead of generating one")
    args = ap.parse_args(argv)

    out = Path(args.out).resolve()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        summary = json.loads((run_dir / "summary.json").read_text()) if (run_dir / "summary.json").exists() else {}
    else:
        run_dir = out.parent / "_demo_run"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        res = generate_demo_run(run_dir)
        summary = res.get("summary", {})

    generate_3d_views(run_dir)
    assemble_site(out, run_dir, summary)

    print(f"[build_pages_site] site ready at: {out}")
    print(f"[build_pages_site] landing: {out/'index.html'}  demo: {out/'demo'/'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
