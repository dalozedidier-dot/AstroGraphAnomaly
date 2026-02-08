#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Showcase Plots Suite (top visual output)

Objectif:
- Reproduire une suite complète de plots (type "téléchargement (19)") à partir de `scored.csv`.
- Stabiliser la lecture du score: produire `anomaly_score_hi` (plus grand = plus anomal) + `anomaly_score_norm` [0..1].
- Optionnel: enrichir avec métriques graphe depuis `graph_full.graphml` (degree, clustering, kcore, betweenness approx, community_id).
- Sorties: ~18 PNG dans un dossier unique (par défaut: <out>/plots).

Usage:
  python tools/showcase_plots.py --scored results/run/scored.csv --graph results/run/graph_full.graphml --out results/run --top-k 30
  python tools/showcase_plots.py --scored results/run/scored.csv --out results/run --top-k 30

Flags:
  --plots-dir NAME                 (default "plots")
  --copy-to-screenshots PATH       (copie 6 PNG "README-ready" vers screenshots/)
  --write-enriched                 (écrit scored_enriched.csv + graph_metrics.json si graph présent)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # CI-safe
import matplotlib.pyplot as plt


# -------------------------
# Score normalization (stable)
# -------------------------
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

    # unify direction: anomaly_score_hi (higher => more anomalous)
    s = df["anomaly_score"].to_numpy(float)
    y = df["anomaly_label"].to_numpy(int)
    if np.mean(s[y == -1]) < np.mean(s[y == 1]):
        df["anomaly_score_hi"] = -df["anomaly_score"]
    else:
        df["anomaly_score_hi"] = df["anomaly_score"]

    df["anomaly_score_norm"] = robust_unit_interval(df["anomaly_score_hi"].to_numpy(float))

    return df


# -------------------------
# Graph enrichment (optional)
# -------------------------
def enrich_from_graph(df: pd.DataFrame, graph_path: Path) -> Tuple[pd.DataFrame, dict]:
    import networkx as nx

    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    n = G.number_of_nodes()
    m = G.number_of_edges()

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    try:
        kcore = nx.core_number(G)
    except Exception:
        kcore = {k: 0 for k in G.nodes()}

    # betweenness approx
    try:
        k = min(200, max(10, int(0.03 * max(n, 1))))
        betw = nx.betweenness_centrality(G, k=k, seed=42, normalized=True)
    except Exception:
        betw = {k: 0.0 for k in G.nodes()}

    # communities
    comm_id: Dict[str, int] = {}
    comm_sizes: List[int] = []
    comm_algo = "none"
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
        comm_algo = "louvain"
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G)
            comm_algo = "greedy_modularity"
        except Exception:
            comms = [set(G.nodes())]
            comm_algo = "single"

    for i, cset in enumerate(comms):
        for node in cset:
            comm_id[str(node)] = i
    comm_sizes = [len(c) for c in comms]

    df2 = df.copy()
    if "source_id" in df2.columns:
        sid = df2["source_id"].astype(str)
        df2["degree"] = sid.map(degree).fillna(0).astype(int)
        df2["clustering"] = sid.map(clustering).fillna(0.0).astype(float)
        df2["kcore"] = sid.map(kcore).fillna(0).astype(int)
        df2["betweenness"] = sid.map(betw).fillna(0.0).astype(float)
        df2["community_id"] = sid.map(comm_id).fillna(-1).astype(int)

    metrics = {
        "n_nodes": int(n),
        "n_edges": int(m),
        "n_components": int(nx.number_connected_components(G)) if n > 0 else 0,
        "comm_algo": comm_algo,
        "n_communities": int(len(comm_sizes)),
        "community_sizes_top10": sorted(comm_sizes, reverse=True)[:10],
    }
    return df2, metrics


# -------------------------
# Plot helpers
# -------------------------
def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig_text(msg: str):
    fig = plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, msg, ha="center", va="center")
    plt.axis("off")
    return fig


# -------------------------
# Plots (curated + extras)
# -------------------------
def plot_score_hist(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    plt.hist(df["anomaly_score_hi"].to_numpy(float), bins=60, alpha=0.85)
    plt.title("Distribution des Scores d'Anomalie (high=anomal)")
    plt.xlabel("anomaly_score_hi")
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    return fig


def plot_ra_dec_score(df: pd.DataFrame):
    if "ra" not in df.columns or "dec" not in df.columns:
        return fig_text("Missing ra/dec")
    fig = plt.figure(figsize=(10, 6))
    ra = pd.to_numeric(df["ra"], errors="coerce").fillna(0.0).to_numpy(float)
    dec = pd.to_numeric(df["dec"], errors="coerce").fillna(0.0).to_numpy(float)
    sc = plt.scatter(ra, dec, s=25, alpha=0.85, c=df["anomaly_score_norm"].to_numpy(float))
    plt.title("RA vs Dec (colored by anomaly score)")
    plt.xlabel("RA [deg]")
    plt.ylabel("Dec [deg]")
    plt.colorbar(sc, label="anomaly_score_norm")
    plt.grid(alpha=0.25)
    return fig


def plot_top_bar(df: pd.DataFrame, top_k: int):
    fig = plt.figure(figsize=(14, 6))
    d = df.sort_values("anomaly_score_hi", ascending=False).head(top_k).copy()
    labels = d["source_id"].astype(str).tolist() if "source_id" in d.columns else [str(i) for i in range(len(d))]
    y = d["anomaly_score_hi"].to_numpy(float)
    plt.bar(range(len(d)), y, alpha=0.9)
    plt.title(f"Top-{top_k} anomalies (anomaly_score_hi)")
    plt.xlabel("Source ID")
    plt.ylabel("anomaly_score_hi")
    plt.xticks(range(len(d)), labels, rotation=90)
    plt.grid(axis="y", alpha=0.25)
    return fig


def plot_mean_features(df: pd.DataFrame):
    fig = plt.figure(figsize=(14, 7))
    df = df.copy()
    df["group"] = np.where(df["anomaly_label"].to_numpy(int) == -1, "Anomalous", "Normal")

    candidates = [
        "phot_g_mean_mag", "bp_rp", "parallax", "pmra", "pmdec", "distance",
        "degree", "clustering", "kcore", "betweenness"
    ]
    feats = [c for c in candidates if c in df.columns]
    if len(feats) == 0:
        return fig_text("No comparable numeric features found")

    m = df.groupby("group")[feats].mean(numeric_only=True)
    x = np.arange(len(feats))
    width = 0.38
    plt.bar(x - width/2, m.loc["Anomalous"].to_numpy(float), width, label="Anomalous", alpha=0.85)
    plt.bar(x + width/2, m.loc["Normal"].to_numpy(float), width, label="Normal", alpha=0.85)
    plt.xticks(x, feats)
    plt.ylabel("mean")
    plt.title("Mean features: Anomalous vs Normal")
    plt.legend()
    plt.grid(axis="y", alpha=0.35)
    return fig


def plot_pca(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 8))
    ignore = {"source_id", "group"}
    numeric_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in ("anomaly_label",)]
    if len(numeric_cols) < 2:
        return fig_text("Not enough numeric columns for PCA")

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :2] * S[:2]

    colors = np.where(df["anomaly_label"].to_numpy(int) == -1, 1.0, 0.0)
    plt.scatter(Z[:, 0], Z[:, 1], s=18, alpha=0.78, c=colors)
    plt.title("PCA 2D (anomalous highlighted)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    return fig


def plot_mag_distance(df: pd.DataFrame):
    if "phot_g_mean_mag" not in df.columns or "distance" not in df.columns:
        return fig_text("Missing phot_g_mean_mag or distance")
    fig = plt.figure(figsize=(10, 6))
    dist = pd.to_numeric(df["distance"], errors="coerce").fillna(0.0).to_numpy(float)
    g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").fillna(0.0).to_numpy(float)
    sc = plt.scatter(dist, g, s=18, alpha=0.78, c=df["anomaly_score_norm"].to_numpy(float))
    plt.gca().invert_yaxis()
    plt.title("G vs Distance (colored by score)")
    plt.xlabel("Distance [pc]")
    plt.ylabel("G [mag]")
    plt.colorbar(sc, label="anomaly_score_norm")
    plt.grid(alpha=0.25)
    return fig


def plot_cmd(df: pd.DataFrame):
    if "bp_rp" not in df.columns or "phot_g_mean_mag" not in df.columns:
        return fig_text("Missing bp_rp or phot_g_mean_mag")
    fig = plt.figure(figsize=(10, 6))
    x = pd.to_numeric(df["bp_rp"], errors="coerce").fillna(0.0).to_numpy(float)
    y = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").fillna(0.0).to_numpy(float)
    plt.scatter(x, y, s=9, alpha=0.75)
    plt.gca().invert_yaxis()
    plt.title("CMD (BP-RP vs G)")
    plt.xlabel("BP-RP [mag]")
    plt.ylabel("G [mag]")
    plt.grid(alpha=0.25)
    return fig


def plot_pm(df: pd.DataFrame):
    if "pmra" not in df.columns or "pmdec" not in df.columns:
        return fig_text("Missing pmra/pmdec")
    fig = plt.figure(figsize=(10, 6))
    x = pd.to_numeric(df["pmra"], errors="coerce").fillna(0.0).to_numpy(float)
    y = pd.to_numeric(df["pmdec"], errors="coerce").fillna(0.0).to_numpy(float)
    sc = plt.scatter(x, y, s=20, alpha=0.75, c=df["anomaly_score_norm"].to_numpy(float))
    plt.title("Proper motion (pmra vs pmdec)")
    plt.xlabel("pmra [mas/yr]")
    plt.ylabel("pmdec [mas/yr]")
    plt.colorbar(sc, label="anomaly_score_norm")
    plt.grid(alpha=0.25)
    return fig


def plot_hist(df: pd.DataFrame, col: str, title: str, bins: int = 60):
    if col not in df.columns:
        return fig_text(f"Missing {col}")
    fig = plt.figure(figsize=(10, 6))
    x = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    plt.hist(x, bins=bins, alpha=0.85)
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("count")
    plt.grid(alpha=0.25)
    return fig


def plot_score_vs(df: pd.DataFrame, col: str, title: str):
    if col not in df.columns:
        return fig_text(f"Missing {col}")
    fig = plt.figure(figsize=(10, 6))
    x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)
    y = df["anomaly_score_hi"].to_numpy(float)
    plt.scatter(x, y, s=18, alpha=0.6, c=df["anomaly_score_norm"].to_numpy(float))
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel("anomaly_score_hi")
    plt.grid(alpha=0.25)
    return fig


def plot_graph_communities(df: pd.DataFrame, graph_path: Path):
    import networkx as nx

    if not graph_path.exists():
        return fig_text("Missing graph_full.graphml")

    fig = plt.figure(figsize=(11, 9))
    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    # pos: ra/dec if available for most nodes, else spring
    pos = {}
    if "ra" in df.columns and "dec" in df.columns and "source_id" in df.columns:
        tmp = df.set_index("source_id")
        for n in G.nodes():
            if n in tmp.index:
                pos[n] = (float(tmp.loc[n, "ra"]), float(tmp.loc[n, "dec"]))
    if len(pos) != G.number_of_nodes():
        pos = nx.spring_layout(G, seed=42)

    # color by community_id if exists, else by degree
    if "community_id" in df.columns and "source_id" in df.columns:
        cmap = dict(zip(df["source_id"].astype(str).tolist(), df["community_id"].to_numpy(int)))
        node_color = np.array([cmap.get(str(n), -1) for n in G.nodes()], dtype=float)
        cbar_label = "community_id"
    else:
        deg = dict(G.degree())
        node_color = np.array([deg.get(str(n), 0) for n in G.nodes()], dtype=float)
        cbar_label = "degree"

    # size by score norm
    score_map = {}
    if "source_id" in df.columns and "anomaly_score_norm" in df.columns:
        score_map = dict(zip(df["source_id"].astype(str).tolist(), df["anomaly_score_norm"].to_numpy(float)))
    sizes = [40 + 140 * score_map.get(str(n), 0.2) for n in G.nodes()]

    nodes = list(G.nodes())
    pc = nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_size=sizes, node_color=node_color, alpha=0.92)
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=0.6)
    plt.title("k-NN graph view (communities/degree + anomaly size)")
    plt.axis("off")
    plt.colorbar(pc, label=cbar_label)
    return fig


def plot_community_sizes(df: pd.DataFrame):
    if "community_id" not in df.columns:
        return fig_text("Missing community_id")
    fig = plt.figure(figsize=(12, 6))
    counts = df["community_id"].value_counts().sort_values(ascending=False).head(25)
    plt.bar(range(len(counts)), counts.to_numpy(int), alpha=0.9)
    plt.title("Top community sizes")
    plt.xlabel("community rank")
    plt.ylabel("size")
    plt.grid(axis="y", alpha=0.25)
    return fig


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--graph", default="", help="Path to graph_full.graphml (optional)")
    ap.add_argument("--out", required=True, help="Run output dir (contains scored.csv)")
    ap.add_argument("--plots-dir", default="plots", help="Subdir under --out for PNGs")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--write-enriched", action="store_true", help="Write scored_enriched.csv + graph_metrics.json if graph present")
    ap.add_argument("--copy-to-screenshots", default="", help="Copy 6 curated PNG to this folder (e.g. screenshots/)")
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    plots_out = out / args.plots_dir
    plots_out.mkdir(parents=True, exist_ok=True)

    scored = Path(args.scored)
    df = pd.read_csv(scored)
    df = ensure_core(df)

    graph_metrics = None
    graph_path = Path(args.graph) if args.graph else None
    if graph_path is not None and graph_path.exists():
        df, graph_metrics = enrich_from_graph(df, graph_path)
        if args.write_enriched:
            df.to_csv(out / "scored_enriched.csv", index=False)
            (out / "graph_metrics.json").write_text(json.dumps(graph_metrics, indent=2), encoding="utf-8")

    # suite ~18 PNG
    suite = [
        ("score_hist.png", plot_score_hist(df)),
        ("ra_dec_score.png", plot_ra_dec_score(df)),
        ("mag_vs_distance.png", plot_mag_distance(df)),
        ("cmd_bp_rp_vs_g.png", plot_cmd(df)),
        ("pca_2d.png", plot_pca(df)),
        ("top_anomalies_scores.png", plot_top_bar(df, top_k=args.top_k)),
        ("mean_features_anom_vs_normal.png", plot_mean_features(df)),
        ("pmra_pmdec_score.png", plot_pm(df)),
        ("parallax_hist.png", plot_hist(df, "parallax", "Parallax distribution")),
        ("distance_hist.png", plot_hist(df, "distance", "Distance distribution")),
        ("degree_hist.png", plot_hist(df, "degree", "Degree distribution (graph)")),
        ("kcore_hist.png", plot_hist(df, "kcore", "k-core distribution (graph)")),
        ("betweenness_hist.png", plot_hist(df, "betweenness", "Betweenness distribution (approx)")),
        ("score_vs_degree.png", plot_score_vs(df, "degree", "anomaly_score vs degree")),
        ("score_vs_kcore.png", plot_score_vs(df, "kcore", "anomaly_score vs kcore")),
        ("score_vs_betweenness.png", plot_score_vs(df, "betweenness", "anomaly_score vs betweenness")),
        ("community_sizes.png", plot_community_sizes(df)),
    ]

    if graph_path is not None and graph_path.exists():
        suite.append(("graph_communities_anomalies.png", plot_graph_communities(df, graph_path)))
    else:
        suite.append(("graph_communities_anomalies.png", fig_text("graph_full.graphml not provided")))

    for name, fig in suite:
        save(fig, plots_out, name)

    # Copy curated subset for README/screenshots
    if args.copy_to_screenshots:
        import shutil
        dst = Path(args.copy_to_screenshots)
        dst.mkdir(parents=True, exist_ok=True)
        curated = [
            "graph_communities_anomalies.png",
            "ra_dec_score.png",
            "cmd_bp_rp_vs_g.png",
            "pca_2d.png",
            "top_anomalies_scores.png",
            "mean_features_anom_vs_normal.png",
        ]
        for fn in curated:
            src = plots_out / fn
            if src.exists():
                shutil.copy2(src, dst / fn)

    print(f"OK: wrote {len(suite)} PNG -> {plots_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
