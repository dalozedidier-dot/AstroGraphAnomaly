#!/usr/bin/env python3
"""
AstroGraphAnomaly - Image Report Generator

Generates a set of PNG images + a lightweight HTML index from a run directory.

Designed for Google Colab and GitHub workflows.
No seaborn dependency (matplotlib only).

Typical inputs (depending on what your pipeline produced):
- raw.csv
- scored.csv
- top_anomalies.csv
- scored_enriched.csv (optional)
- graph_full.graphml (optional)

Usage:
  python tools/generate_image_report.py --run-dir results/<run>

Outputs:
  <run-dir>/image_report/
    *.png
    index.html
    report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib in headless mode (CI/Colab)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import networkx as nx  # type: ignore
except Exception:  # pragma: no cover
    nx = None


@dataclass
class Inputs:
    run_dir: Path
    out_dir: Path
    raw_csv: Optional[Path]
    scored_csv: Optional[Path]
    top_csv: Optional[Path]
    enriched_csv: Optional[Path]
    graph_graphml: Optional[Path]


def _safe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if not path or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        # Try with low_memory off
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception:
            return None


def _autodetect_in_run_dir(run_dir: Path) -> Inputs:
    raw = run_dir / "raw.csv"
    scored = run_dir / "scored.csv"
    top = run_dir / "top_anomalies.csv"
    enriched = run_dir / "scored_enriched.csv"
    graph = run_dir / "graph_full.graphml"

    raw_csv = raw if raw.exists() else None
    scored_csv = scored if scored.exists() else None
    top_csv = top if top.exists() else None
    enriched_csv = enriched if enriched.exists() else None
    graph_graphml = graph if graph.exists() else None

    out_dir = run_dir / "image_report"
    return Inputs(
        run_dir=run_dir,
        out_dir=out_dir,
        raw_csv=raw_csv,
        scored_csv=scored_csv,
        top_csv=top_csv,
        enriched_csv=enriched_csv,
        graph_graphml=graph_graphml,
    )


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _col(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    if name in df.columns:
        return df[name]
    # Common alternatives
    alt = {
        "source_id": ["source_id", "SOURCE_ID", "id"],
        "ra": ["ra", "RA"],
        "dec": ["dec", "DEC"],
        "anomaly_score": ["anomaly_score", "score", "anomalyScore"],
        "anomaly_label": ["anomaly_label", "label"],
        "bp_rp": ["bp_rp", "bp_rp_color", "bp_minus_rp"],
        "phot_g_mean_mag": ["phot_g_mean_mag", "g_mag", "phot_g_mean_mag"],
        "parallax": ["parallax"],
        "pmra": ["pmra"],
        "pmdec": ["pmdec"],
        "distance": ["distance", "dist_pc", "dist"],
        "community_id": ["community_id", "community", "louvain"],
    }
    for k, cands in alt.items():
        if k == name:
            for c in cands:
                if c in df.columns:
                    return df[c]
    return None


def _savefig(path: Path, title: str | None = None) -> None:
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _maybe_make_score(df_scored: pd.DataFrame) -> Optional[pd.Series]:
    s = _col(df_scored, "anomaly_score")
    if s is None:
        return None
    return pd.to_numeric(s, errors="coerce")


def _maybe_make_label(df_scored: pd.DataFrame) -> Optional[pd.Series]:
    lab = _col(df_scored, "anomaly_label")
    if lab is None:
        return None
    # Normalize to {1, -1} if possible
    x = pd.to_numeric(lab, errors="coerce")
    if x.isna().all():
        return None
    # Some pipelines may use 0/1
    if set(x.dropna().unique()).issubset({0, 1}):
        return x.map({0: 1, 1: -1})
    return x


def plot_score_hist(df_scored: pd.DataFrame, out_png: Path) -> bool:
    score = _maybe_make_score(df_scored)
    if score is None:
        return False
    score = score.dropna()
    if score.empty:
        return False
    plt.figure(figsize=(10, 6))
    plt.hist(score.values, bins=50)
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    _savefig(out_png, "Score distribution")
    return True


def plot_score_rank_curve(df_scored: pd.DataFrame, out_png: Path) -> bool:
    score = _maybe_make_score(df_scored)
    if score is None:
        return False
    score = score.dropna().sort_values(ascending=False).reset_index(drop=True)
    if score.empty:
        return False
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(score)), score.values)
    plt.xlabel("Rank (descending)")
    plt.ylabel("Anomaly score")
    _savefig(out_png, "Score rank curve")
    return True


def plot_ra_dec_score(df_scored: pd.DataFrame, out_png: Path) -> bool:
    ra = _col(df_scored, "ra")
    dec = _col(df_scored, "dec")
    score = _maybe_make_score(df_scored)
    if ra is None or dec is None or score is None:
        return False
    ra = pd.to_numeric(ra, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")
    score = score
    m = ~(ra.isna() | dec.isna() | score.isna())
    if m.sum() == 0:
        return False
    plt.figure(figsize=(11, 7))
    sc = plt.scatter(ra[m].values, dec[m].values, c=score[m].values, s=60, cmap="viridis", alpha=0.8)
    plt.xlabel("Right Ascension (RA)")
    plt.ylabel("Declination (Dec)")
    plt.colorbar(sc, label="Anomaly Score")
    _savefig(out_png, "Spatial distribution (RA vs Dec) colored by score")
    return True


def plot_cmd(df_scored: pd.DataFrame, out_png: Path) -> bool:
    bp_rp = _col(df_scored, "bp_rp")
    gmag = _col(df_scored, "phot_g_mean_mag")
    if bp_rp is None or gmag is None:
        return False
    bp_rp = pd.to_numeric(bp_rp, errors="coerce")
    gmag = pd.to_numeric(gmag, errors="coerce")
    m = ~(bp_rp.isna() | gmag.isna())
    if m.sum() == 0:
        return False
    plt.figure(figsize=(10, 7))
    plt.scatter(bp_rp[m].values, gmag[m].values, s=6, alpha=0.6)
    plt.gca().invert_yaxis()
    plt.xlabel("BP - RP Color [mag]")
    plt.ylabel("G-band Magnitude [mag]")
    _savefig(out_png, "Gaia Color-Magnitude Diagram (BP-RP vs G)")
    return True


def plot_mean_features_anom_vs_normal(df_scored: pd.DataFrame, out_png: Path) -> bool:
    label = _maybe_make_label(df_scored)
    if label is None:
        return False

    # Choose a small stable set of features if available
    feature_names = ["phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec"]
    cols = []
    for fn in feature_names:
        s = _col(df_scored, fn)
        if s is not None:
            cols.append(fn)
    if not cols:
        return False

    tmp = df_scored.copy()
    tmp["_label"] = pd.to_numeric(label, errors="coerce")
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce")

    # "Anomalous" usually -1
    anom = tmp[tmp["_label"] == -1][cols].mean(numeric_only=True)
    norm = tmp[tmp["_label"] != -1][cols].mean(numeric_only=True)
    if anom.isna().all() or norm.isna().all():
        return False

    x = np.arange(len(cols))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, anom.values, width, label="Anomalous")
    plt.bar(x + width / 2, norm.values, width, label="Normal")
    plt.xticks(x, cols, rotation=30, ha="right")
    plt.ylabel("Mean value")
    plt.legend()
    _savefig(out_png, "Comparison of mean feature values: anomalous vs normal")
    return True


def plot_top_anomalies_bar(df_top: pd.DataFrame, out_png: Path) -> bool:
    sid = _col(df_top, "source_id")
    score = _col(df_top, "anomaly_score")
    if sid is None or score is None:
        return False
    sid = sid.astype(str)
    score = pd.to_numeric(score, errors="coerce")
    m = ~score.isna()
    if m.sum() == 0:
        return False
    # Keep top 30 max
    tmp = pd.DataFrame({"source_id": sid[m], "anomaly_score": score[m]}).sort_values("anomaly_score", ascending=False).head(30)
    plt.figure(figsize=(16, 6))
    plt.bar(tmp["source_id"].values, tmp["anomaly_score"].values)
    plt.xticks(rotation=90)
    plt.xlabel("Source ID (top candidates)")
    plt.ylabel("Anomaly score")
    _savefig(out_png, "Anomaly score per anomalous Source ID")
    return True


def plot_top_anomalies_ra_dec(df_top: pd.DataFrame, out_png: Path) -> bool:
    ra = _col(df_top, "ra")
    dec = _col(df_top, "dec")
    score = _col(df_top, "anomaly_score")
    if ra is None or dec is None or score is None:
        return False
    ra = pd.to_numeric(ra, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")
    score = pd.to_numeric(score, errors="coerce")
    m = ~(ra.isna() | dec.isna() | score.isna())
    if m.sum() == 0:
        return False
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(ra[m].values, dec[m].values, c=score[m].values, s=60, cmap="viridis")
    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.colorbar(sc, label="anomaly_score")
    _savefig(out_png, "Top anomalies: position and score")
    return True


def plot_region_distribution(df_raw: pd.DataFrame, out_png: Path) -> bool:
    ra = _col(df_raw, "ra")
    dec = _col(df_raw, "dec")
    if ra is None or dec is None:
        return False
    ra = pd.to_numeric(ra, errors="coerce")
    dec = pd.to_numeric(dec, errors="coerce")
    m = ~(ra.isna() | dec.isna())
    if m.sum() == 0:
        return False
    plt.figure(figsize=(10, 6))
    plt.scatter(ra[m].values, dec[m].values, s=6, alpha=0.7)
    plt.xlabel("Right Ascension (RA) [deg]")
    plt.ylabel("Declination (Dec) [deg]")
    _savefig(out_png, "Distribution of Gaia sources in the selected region")
    return True


def _load_graph(path: Optional[Path]):
    if path is None or not path.exists() or nx is None:
        return None
    try:
        return nx.read_graphml(path)
    except Exception:
        try:
            # Some GraphML files need node_type cast
            return nx.read_graphml(path, node_type=str)
        except Exception:
            return None


def _scores_map(df_scored: pd.DataFrame) -> Dict[str, float]:
    sid = _col(df_scored, "source_id")
    score = _maybe_make_score(df_scored)
    if sid is None or score is None:
        return {}
    sid = sid.astype(str)
    score = pd.to_numeric(score, errors="coerce")
    m = ~score.isna()
    return dict(zip(sid[m].values, score[m].values))


def plot_graph_nodes_by_score(df_scored: pd.DataFrame, graph, out_png: Path) -> bool:
    if nx is None or graph is None:
        return False

    scores = _scores_map(df_scored)
    if not scores:
        return False

    nodes = list(graph.nodes())
    n = len(nodes)
    if n == 0:
        return False

    # Deterministic layout (can be slow, but acceptable for ~1200 nodes)
    try:
        pos = nx.spring_layout(graph, seed=42, iterations=60)
    except Exception:
        return False

    xs = []
    ys = []
    cs = []
    for node in nodes:
        key = str(node)
        x, y = pos[node]
        xs.append(x)
        ys.append(y)
        cs.append(scores.get(key, np.nan))

    cs_arr = np.array(cs, dtype=float)
    if np.isnan(cs_arr).all():
        return False
    # Replace NaNs with median for display
    med = np.nanmedian(cs_arr)
    cs_arr = np.where(np.isnan(cs_arr), med, cs_arr)

    plt.figure(figsize=(10, 10))
    # Sample edges to keep it fast and readable
    edges = list(graph.edges())
    if len(edges) > 6000:
        random.Random(42).shuffle(edges)
        edges = edges[:6000]
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], linewidth=0.2, alpha=0.15)

    sc = plt.scatter(xs, ys, c=cs_arr, s=12, cmap="viridis", alpha=0.9)
    plt.axis("off")
    plt.colorbar(sc, label="Anomaly Score")
    _savefig(out_png, "Visualization of the generated graph (nodes colored by score)")
    return True


def _community_ids_from_enriched(df_enriched: pd.DataFrame) -> Optional[Dict[str, int]]:
    sid = _col(df_enriched, "source_id")
    cid = _col(df_enriched, "community_id")
    if sid is None or cid is None:
        return None
    sid = sid.astype(str)
    cid = pd.to_numeric(cid, errors="coerce")
    m = ~cid.isna()
    if m.sum() == 0:
        return None
    return dict(zip(sid[m].values, cid[m].astype(int).values))


def _community_ids_compute(graph) -> Optional[Dict[str, int]]:
    if nx is None or graph is None:
        return None
    # Use louvain_communities if available (networkx>=2.8)
    try:
        comms = nx.algorithms.community.louvain_communities(graph, seed=42)
    except Exception:
        try:
            comms = nx.algorithms.community.greedy_modularity_communities(graph)
        except Exception:
            return None
    cid_map: Dict[str, int] = {}
    for i, comm in enumerate(comms):
        for node in comm:
            cid_map[str(node)] = i
    return cid_map or None


def plot_graph_anomalies_by_community(
    df_scored: pd.DataFrame,
    graph,
    df_enriched: Optional[pd.DataFrame],
    out_png: Path,
) -> bool:
    if nx is None or graph is None:
        return False

    scores = _scores_map(df_scored)
    if not scores:
        return False

    cid_map = None
    if df_enriched is not None:
        cid_map = _community_ids_from_enriched(df_enriched)
    if cid_map is None:
        cid_map = _community_ids_compute(graph)
    if cid_map is None:
        return False

    # Layout
    try:
        pos = nx.spring_layout(graph, seed=42, iterations=60)
    except Exception:
        return False

    nodes = list(graph.nodes())
    xs, ys, cs = [], [], []
    for node in nodes:
        x, y = pos[node]
        xs.append(x)
        ys.append(y)
        cs.append(cid_map.get(str(node), -1))

    cs_arr = np.array(cs, dtype=float)
    plt.figure(figsize=(11, 11))
    edges = list(graph.edges())
    if len(edges) > 6000:
        random.Random(42).shuffle(edges)
        edges = edges[:6000]
    for u, v in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], linewidth=0.2, alpha=0.15)

    sc = plt.scatter(xs, ys, c=cs_arr, s=14, cmap="tab20", alpha=0.95)
    plt.axis("off")
    plt.colorbar(sc, label="Community ID")
    _savefig(out_png, "Graph anomalies by community (k-NN)")
    return True


def plot_graph_anomalies_knn_subgraph(df_top: pd.DataFrame, graph, out_png: Path) -> bool:
    if nx is None or graph is None:
        return False

    sid = _col(df_top, "source_id")
    score = _col(df_top, "anomaly_score")
    if sid is None or score is None:
        return False

    top_nodes = [str(x) for x in sid.astype(str).values[:30]]
    # Expand with one-hop neighbors for context
    keep = set()
    for n in top_nodes:
        if n in graph:
            keep.add(n)
            keep.update(list(graph.neighbors(n)))
    if not keep:
        return False

    sg = graph.subgraph(list(keep)).copy()
    if sg.number_of_nodes() == 0:
        return False

    # Layout
    try:
        pos = nx.spring_layout(sg, seed=42, iterations=80)
    except Exception:
        return False

    # Node colors by score if present
    score_map = {str(s): float(v) for s, v in zip(sid.astype(str).values, pd.to_numeric(score, errors="coerce").values) if not pd.isna(v)}
    node_c = []
    xs, ys = [], []
    for n in sg.nodes():
        x, y = pos[n]
        xs.append(x)
        ys.append(y)
        node_c.append(score_map.get(str(n), np.nan))
    cs = np.array(node_c, dtype=float)
    if np.isnan(cs).all():
        cs = np.zeros_like(cs)
    med = np.nanmedian(cs)
    cs = np.where(np.isnan(cs), med, cs)

    plt.figure(figsize=(9, 9))
    for u, v in sg.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        plt.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.6)

    sc = plt.scatter(xs, ys, c=cs, s=90, cmap="viridis")
    plt.axis("off")
    plt.colorbar(sc, label="Anomaly Score")
    _savefig(out_png, "Graph anomalies (k-NN)")
    return True


def _build_html_index(out_dir: Path, generated: List[str], stats: Dict[str, Any]) -> None:
    items = "\n".join(
        f'<div class="card"><a href="{name}"><img src="{name}" alt="{name}"/></a><div class="cap">{name}</div></div>'
        for name in generated
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>AstroGraphAnomaly - Image Report</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ margin: 0 0 8px 0; }}
.meta {{ color: #444; margin-bottom: 18px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
.card {{ border: 1px solid #ddd; border-radius: 10px; padding: 10px; background: #fff; }}
.card img {{ width: 100%; height: auto; border-radius: 8px; }}
.cap {{ font-size: 12px; color: #333; margin-top: 6px; word-break: break-all; }}
pre {{ background: #f6f8fa; padding: 10px; border-radius: 8px; overflow-x: auto; }}
</style>
</head>
<body>
<h1>AstroGraphAnomaly - Image Report</h1>
<div class="meta">Generated in <code>{out_dir}</code></div>
<h2>Quick stats</h2>
<pre>{json.dumps(stats, indent=2)}</pre>
<h2>Images</h2>
<div class="grid">
{items}
</div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def generate(inputs: Inputs) -> Dict[str, Any]:
    _ensure_dir(inputs.out_dir)

    df_raw = _safe_read_csv(inputs.raw_csv)
    df_scored = _safe_read_csv(inputs.scored_csv)
    df_top = _safe_read_csv(inputs.top_csv)
    df_enriched = _safe_read_csv(inputs.enriched_csv)
    graph = _load_graph(inputs.graph_graphml)

    generated: List[str] = []
    def _do(fn, name: str, *args) -> None:
        out = inputs.out_dir / name
        ok = fn(*args, out)
        if ok:
            generated.append(name)

    stats: Dict[str, Any] = {"run_dir": str(inputs.run_dir), "generated_count": 0}
    if df_scored is not None:
        stats["n_scored"] = int(len(df_scored))
    if df_raw is not None:
        stats["n_raw"] = int(len(df_raw))
    if df_top is not None:
        stats["n_top"] = int(len(df_top))
    if graph is not None:
        stats["graph_nodes"] = int(graph.number_of_nodes())
        stats["graph_edges"] = int(graph.number_of_edges())

    if df_scored is not None:
        _do(plot_graph_nodes_by_score, "01_graph_nodes_by_score.png", df_scored, graph)
        _do(plot_ra_dec_score, "02_ra_dec_score.png", df_scored)
        _do(plot_mean_features_anom_vs_normal, "03_mean_features_anom_vs_normal.png", df_scored)
        _do(plot_score_hist, "10_score_hist.png", df_scored)
        _do(plot_score_rank_curve, "11_score_rank_curve.png", df_scored)
        _do(plot_cmd, "07_cmd_bp_rp_vs_g.png", df_scored)

    if df_raw is not None:
        _do(plot_region_distribution, "08_region_distribution.png", df_raw)

    if df_top is not None:
        _do(plot_top_anomalies_bar, "04_top_anomalies_scores.png", df_top)
        _do(plot_top_anomalies_ra_dec, "06_top_anomalies_position_score.png", df_top)
        _do(plot_graph_anomalies_knn_subgraph, "05_graph_anomalies_knn.png", df_top, graph)

    if df_scored is not None:
        _do(plot_graph_anomalies_by_community, "09_graph_anomalies_by_community.png", df_scored, graph, df_enriched)

    stats["generated_count"] = len(generated)

    report = {"stats": stats, "images": generated}
    (inputs.out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _build_html_index(inputs.out_dir, generated, stats)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="Path to a run directory (contains scored.csv etc.)")
    ap.add_argument("--out-dir", default=None, help="Override output directory (default: <run-dir>/image_report)")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir).expanduser().resolve()
    inputs = _autodetect_in_run_dir(run_dir)
    if args.out_dir:
        inputs.out_dir = Path(args.out_dir).expanduser().resolve()

    report = generate(inputs)
    print(json.dumps(report["stats"], indent=2))
    print(f"Wrote: {inputs.out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
