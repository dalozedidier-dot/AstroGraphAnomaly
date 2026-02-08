#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Post-analyse (extended)

Écrit dans <out_dir>/analysis/ :
- graph_metrics.csv
- community_sizes.csv
- plots/*.png (diagnostics)
- summary.json

But: compléter la sortie du workflow avec des métriques graphe avancées + visuels diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


def post_analyze(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)
    analysis_dir = out_dir / "analysis"
    plots_dir = analysis_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scored_path = out_dir / "scored.csv"
    graph_path = out_dir / "graph_full.graphml"
    if not scored_path.exists() or not graph_path.exists():
        return {"status": "skipped_missing_inputs"}

    df = pd.read_csv(scored_path)
    df["source_id"] = df["source_id"].astype(str)

    G = nx.read_graphml(graph_path)
    nodes = list(G.nodes())

    deg = dict(G.degree())
    clust = nx.clustering(G)
    core = nx.core_number(G) if G.number_of_nodes() > 0 else {n: 0 for n in nodes}

    # Betweenness approx (k)
    k = min(300, len(nodes))
    btw = nx.betweenness_centrality(G, k=k, normalized=True, seed=42) if len(nodes) > 1 else {n: 0.0 for n in nodes}

    # Communities (Louvain if available, else greedy modularity)
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
        comm_algo = "louvain"
    except Exception:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = greedy_modularity_communities(G)
        comm_algo = "greedy_modularity"

    comm_id: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            comm_id[str(n)] = int(i)

    # Articulation points + bridges
    aps = set(str(x) for x in nx.articulation_points(G)) if G.number_of_nodes() > 2 else set()
    try:
        bridges = list(nx.bridges(G))
        bridge_nodes = set([str(a) for a, b in bridges] + [str(b) for a, b in bridges])
    except Exception:
        bridge_nodes = set()

    gdf = pd.DataFrame({
        "source_id": [str(n) for n in nodes],
        "degree": [deg.get(n, 0) for n in nodes],
        "clustering": [clust.get(n, 0.0) for n in nodes],
        "kcore": [core.get(n, 0) for n in nodes],
        "betweenness": [btw.get(n, 0.0) for n in nodes],
        "community": [comm_id.get(str(n), -1) for n in nodes],
        "is_articulation": [1 if str(n) in aps else 0 for n in nodes],
        "incident_to_bridge": [1 if str(n) in bridge_nodes else 0 for n in nodes],
    })

    cols = ["source_id"]
    if "anomaly_score" in df.columns:
        cols.append("anomaly_score")
    if "anomaly_label" in df.columns:
        cols.append("anomaly_label")

    m = gdf.merge(df[cols], on="source_id", how="left")

    (analysis_dir / "graph_metrics.csv").write_text(m.to_csv(index=False), encoding="utf-8")

    cs = m.groupby("community").size().sort_values(ascending=False).reset_index(name="size")
    (analysis_dir / "community_sizes.csv").write_text(cs.to_csv(index=False), encoding="utf-8")

    # Diagnostics plots
    plt.figure(figsize=(9, 5.5))
    plt.hist(m["degree"].to_numpy(int), bins=50, alpha=0.85)
    plt.title("Degree distribution")
    plt.xlabel("degree"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "degree_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.hist(m["kcore"].to_numpy(int), bins=30, alpha=0.85)
    plt.title("k-core distribution")
    plt.xlabel("kcore"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "kcore_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5.5))
    plt.hist(m["betweenness"].to_numpy(float), bins=60, alpha=0.85)
    plt.title("Betweenness distribution (approx)")
    plt.xlabel("betweenness"); plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "betweenness_hist.png", dpi=160)
    plt.close()

    if "anomaly_score" in m.columns:
        for col in ("degree", "kcore", "betweenness"):
            plt.figure(figsize=(7, 5.5))
            plt.scatter(m[col].to_numpy(float), m["anomaly_score"].to_numpy(float), s=10, alpha=0.55)
            plt.title(f"anomaly_score vs {col}")
            plt.xlabel(col); plt.ylabel("anomaly_score")
            plt.tight_layout()
            plt.savefig(plots_dir / f"score_vs_{col}.png", dpi=160)
            plt.close()

    top = cs.head(25)
    plt.figure(figsize=(10, 5.5))
    plt.bar(range(len(top)), top["size"].to_numpy(int), alpha=0.9)
    plt.title(f"Top community sizes ({comm_algo})")
    plt.xlabel("community rank"); plt.ylabel("size")
    plt.tight_layout()
    plt.savefig(plots_dir / "community_sizes.png", dpi=160)
    plt.close()

    summary = {
        "status": "ok",
        "comm_algo": comm_algo,
        "n_nodes": int(G.number_of_nodes()),
        "n_edges": int(G.number_of_edges()),
        "n_communities": int(len(cs)),
        "n_articulation_points": int(len(aps)),
        "n_bridge_nodes": int(len(bridge_nodes)),
    }
    (analysis_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
