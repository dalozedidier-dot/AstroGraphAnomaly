#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Full Plots Suite (workflow-first, no packaging required)

Entrées:
  - scored.csv (obligatoire)
  - graph_full.graphml (optionnel mais recommandé)

Sorties (plots "curated", contrat stable):
  - score_hist.png
  - ra_dec_score.png
  - top_anomalies_scores.png
  - mean_features_anom_vs_normal.png
  - pca_2d.png
  - mag_vs_distance.png
  - graph_communities_anomalies.png
  - cmd_bp_rp_vs_g.png

Sorties additionnelles (si possible):
  - graph_metrics.json
  - scored_enriched.csv (si graph fourni et métriques calculées)

Usage:
  python tools/full_plots_suite.py \
    --scored results/run/scored.csv \
    --graph  results/run/graph_full.graphml \
    --out    results/run/plots \
    --top-k  30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # CI-safe
import matplotlib.pyplot as plt


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


def ensure_core_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "source_id" in df.columns:
        df["source_id"] = df["source_id"].astype(str)

    if "anomaly_score" not in df.columns:
        df["anomaly_score"] = 0.0
    df["anomaly_score"] = pd.to_numeric(df["anomaly_score"], errors="coerce").fillna(0.0)

    if "anomaly_label" not in df.columns:
        thr = np.nanpercentile(df["anomaly_score"].to_numpy(float), 95)
        df["anomaly_label"] = np.where(df["anomaly_score"].to_numpy(float) >= thr, -1, 1)

    df["anomaly_score_norm"] = robust_unit_interval(df["anomaly_score"].to_numpy(float))
    return df


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / name, dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_score_hist(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    plt.hist(df["anomaly_score"].to_numpy(float), bins=50, alpha=0.85)
    plt.title("Distribution des Scores d'Anomalie")
    plt.xlabel("Score d'Anomalie")
    plt.ylabel("Fréquence")
    plt.grid(alpha=0.25)
    return fig


def plot_ra_dec_score(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    if "ra" in df.columns and "dec" in df.columns:
        ra = pd.to_numeric(df["ra"], errors="coerce").fillna(0.0).to_numpy(float)
        dec = pd.to_numeric(df["dec"], errors="coerce").fillna(0.0).to_numpy(float)
        sc = plt.scatter(ra, dec, s=35, alpha=0.85, c=df["anomaly_score_norm"].to_numpy(float))
        plt.title("Spatial Distribution of Nodes (RA vs Dec) colored by Score")
        plt.xlabel("Right Ascension (RA)")
        plt.ylabel("Declination (Dec)")
        plt.colorbar(sc, label="Anomaly Score (norm)")
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "Missing ra/dec columns", ha="center", va="center")
        plt.axis("off")
    return fig


def plot_top_bar(df: pd.DataFrame, top_k: int):
    fig = plt.figure(figsize=(14, 6))
    d = df.sort_values("anomaly_score", ascending=False).head(top_k).copy()

    labels = d["source_id"].astype(str).tolist() if "source_id" in d.columns else [str(i) for i in range(len(d))]
    y = d["anomaly_score"].to_numpy(float)

    plt.bar(range(len(d)), y, alpha=0.9)
    plt.title("Anomaly Score per Anomalous Source ID (Top-K)")
    plt.xlabel("Source ID (Anomalous Candidates)")
    plt.ylabel("Anomaly Score")
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
        plt.text(0.5, 0.5, "No comparable features found", ha="center", va="center")
        plt.axis("off")
        return fig

    m = df.groupby("group")[feats].mean(numeric_only=True)
    x = np.arange(len(feats))
    width = 0.38

    plt.bar(x - width/2, m.loc["Anomalous"].to_numpy(float), width, label="Anomalous", alpha=0.85)
    plt.bar(x + width/2, m.loc["Normal"].to_numpy(float), width, label="Normal", alpha=0.85)
    plt.xticks(x, feats)
    plt.ylabel("Mean Value")
    plt.title("Comparison of Mean Feature Values: Anomalous vs. Normal")
    plt.legend()
    plt.grid(axis="y", alpha=0.35)
    return fig


def plot_pca(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 8))
    ignore = {"source_id", "group"}
    numeric_cols = [c for c in df.columns if c not in ignore and pd.api.types.is_numeric_dtype(df[c])]
    numeric_cols = [c for c in numeric_cols if c not in ("anomaly_label",)]

    if len(numeric_cols) < 2:
        plt.text(0.5, 0.5, "Not enough numeric columns for PCA", ha="center", va="center")
        plt.axis("off")
        return fig

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    U, S, _ = np.linalg.svd(X, full_matrices=False)
    Z = U[:, :2] * S[:2]

    colors = np.where(df["anomaly_label"].to_numpy(int) == -1, 1.0, 0.0)
    plt.scatter(Z[:, 0], Z[:, 1], s=22, alpha=0.78, c=colors)
    plt.title("PCA 2D (colored: anomalous vs normal)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(alpha=0.25)
    return fig


def plot_mag_distance(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    if "phot_g_mean_mag" in df.columns and "distance" in df.columns:
        dist = pd.to_numeric(df["distance"], errors="coerce").fillna(0.0).to_numpy(float)
        g = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").fillna(0.0).to_numpy(float)
        sc = plt.scatter(dist, g, s=22, alpha=0.75, c=df["anomaly_score_norm"].to_numpy(float))
        plt.gca().invert_yaxis()
        plt.title("G magnitude vs Distance (colored by score)")
        plt.xlabel("Distance (pc)")
        plt.ylabel("G-band magnitude")
        plt.colorbar(sc, label="Anomaly Score (norm)")
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "Missing phot_g_mean_mag or distance", ha="center", va="center")
        plt.axis("off")
    return fig


def plot_cmd(df: pd.DataFrame):
    fig = plt.figure(figsize=(10, 6))
    if "bp_rp" in df.columns and "phot_g_mean_mag" in df.columns:
        x = pd.to_numeric(df["bp_rp"], errors="coerce").fillna(0.0).to_numpy(float)
        y = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").fillna(0.0).to_numpy(float)
        plt.scatter(x, y, s=10, alpha=0.75)
        plt.gca().invert_yaxis()
        plt.title("Diagramme Couleur-Magnitude Gaia (BP-RP vs G)")
        plt.xlabel("BP - RP Color [mag]")
        plt.ylabel("G-band Magnitude [mag]")
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "Missing bp_rp or phot_g_mean_mag", ha="center", va="center")
        plt.axis("off")
    return fig


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

    try:
        k = min(200, max(10, int(0.03 * n)))
        betw = nx.betweenness_centrality(G, k=k, seed=42, normalized=True)
    except Exception:
        betw = {k: 0.0 for k in G.nodes()}

    try:
        arts = list(nx.articulation_points(G))
    except Exception:
        arts = []
    try:
        bridges = [(str(a), str(b)) for a, b in nx.bridges(G)]
    except Exception:
        bridges = []

    comm_id: Dict[str, int] = {}
    comm_sizes: List[int] = []
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
        for i, cset in enumerate(comms):
            for node in cset:
                comm_id[str(node)] = i
        comm_sizes = [len(c) for c in comms]
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G)
            for i, cset in enumerate(comms):
                for node in cset:
                    comm_id[str(node)] = i
            comm_sizes = [len(c) for c in comms]
        except Exception:
            comm_id = {str(node): 0 for node in G.nodes()}
            comm_sizes = [n]

    df2 = df.copy()
    if "source_id" in df2.columns:
        sid = df2["source_id"].astype(str)
        df2["degree"] = sid.map(degree).fillna(0).astype(int)
        df2["clustering"] = sid.map(clustering).fillna(0.0).astype(float)
        df2["kcore"] = sid.map(kcore).fillna(0).astype(int)
        df2["betweenness"] = sid.map(betw).fillna(0.0).astype(float)
        df2["community_id"] = sid.map(comm_id).fillna(-1).astype(int)

    metrics = {
        "n_nodes": n,
        "n_edges": m,
        "n_components": int(nx.number_connected_components(G)) if n > 0 else 0,
        "n_articulation_points": len(arts),
        "n_bridges": len(bridges),
        "articulation_points_sample": [str(x) for x in arts[:50]],
        "bridges_sample": bridges[:50],
        "n_communities": len(comm_sizes),
        "community_sizes_top10": sorted(comm_sizes, reverse=True)[:10],
    }
    return df2, metrics


def plot_graph_communities(df: pd.DataFrame, graph_path: Path):
    import networkx as nx

    fig = plt.figure(figsize=(11, 9))
    if not graph_path.exists():
        plt.text(0.5, 0.5, "Missing graph_full.graphml", ha="center", va="center")
        plt.axis("off")
        return fig

    G = nx.read_graphml(graph_path)
    G = nx.relabel_nodes(G, {n: str(n) for n in G.nodes()})

    pos = {}
    if "ra" in df.columns and "dec" in df.columns and "source_id" in df.columns:
        tmp = df.set_index("source_id")
        for n in G.nodes():
            if n in tmp.index:
                pos[n] = (float(tmp.loc[n, "ra"]), float(tmp.loc[n, "dec"]))
    if len(pos) != G.number_of_nodes():
        pos = nx.spring_layout(G, seed=42)

    if "community_id" in df.columns and "source_id" in df.columns:
        cmap = dict(zip(df["source_id"].astype(str).tolist(), df["community_id"].to_numpy(int)))
        node_color = np.array([cmap.get(str(n), -1) for n in G.nodes()], dtype=float)
    else:
        node_color = np.zeros(G.number_of_nodes(), dtype=float)

    score_map = {}
    if "source_id" in df.columns and "anomaly_score_norm" in df.columns:
        score_map = dict(zip(df["source_id"].astype(str).tolist(), df["anomaly_score_norm"].to_numpy(float)))
    sizes = [40 + 120*score_map.get(str(n), 0.2) for n in G.nodes()]

    nx.draw(G, pos, node_size=sizes, with_labels=False, node_color=node_color, alpha=0.92, width=0.6)
    plt.title("Graphe Anomalies par Communauté (k-NN)")
    plt.colorbar(plt.cm.ScalarMappable(), label="Community ID")
    return fig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--graph", default="", help="Path to graph_full.graphml (optional)")
    ap.add_argument("--out", required=True, help="Output directory for PNGs")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--write-enriched", action="store_true", help="Write scored_enriched.csv + graph_metrics.json")
    return ap.parse_args()


def main():
    args = parse_args()
    scored = Path(args.scored)
    out = Path(args.out)
    graph = Path(args.graph) if args.graph else None

    df = pd.read_csv(scored)
    df = ensure_core_columns(df)

    if graph is not None and graph.exists():
        df, metrics = enrich_from_graph(df, graph)
        if args.write_enriched:
            (out.parent / "graph_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            df.to_csv(out.parent / "scored_enriched.csv", index=False)

    curated = [
        ("score_hist.png", plot_score_hist(df)),
        ("ra_dec_score.png", plot_ra_dec_score(df)),
        ("top_anomalies_scores.png", plot_top_bar(df, top_k=args.top_k)),
        ("mean_features_anom_vs_normal.png", plot_mean_features(df)),
        ("pca_2d.png", plot_pca(df)),
        ("mag_vs_distance.png", plot_mag_distance(df)),
        ("cmd_bp_rp_vs_g.png", plot_cmd(df)),
    ]
    for name, fig in curated:
        save(fig, out, name)

    if graph is not None and graph.exists():
        fig = plot_graph_communities(df, graph)
    else:
        fig = plt.figure(figsize=(11, 9))
        plt.text(0.5, 0.5, "graph_full.graphml not provided", ha="center", va="center")
        plt.axis("off")
    save(fig, out, "graph_communities_anomalies.png")

    print(f"OK: wrote curated plots -> {out}")


if __name__ == "__main__":
    main()
