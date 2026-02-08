from __future__ import annotations
import os
from pathlib import Path
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

def save_basic_plots(out_dir: str, df_scored: pd.DataFrame, anomalies_col: str = "anomaly_label"):
    plot_dir = Path(out_dir)/"plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Score histogram
    plt.figure(figsize=(8,5))
    df_scored["anomaly_score"].hist(bins=60)
    plt.title("Anomaly score distribution (higher = more anomalous)")
    plt.xlabel("anomaly_score")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plot_dir/"score_hist.png", dpi=140)
    plt.close()

    # Mag vs distance
    if "phot_g_mean_mag" in df_scored.columns and "distance" in df_scored.columns:
        plt.figure(figsize=(8,5))
        is_anom = df_scored[anomalies_col].values == -1
        plt.scatter(df_scored.loc[~is_anom, "distance"], df_scored.loc[~is_anom, "phot_g_mean_mag"], s=6, alpha=0.6)
        plt.scatter(df_scored.loc[is_anom, "distance"], df_scored.loc[is_anom, "phot_g_mean_mag"], s=18, alpha=0.9)
        plt.title("Magnitude vs distance (anomalies highlighted)")
        plt.xlabel("distance (pc)")
        plt.ylabel("phot_g_mean_mag")
        plt.tight_layout()
        plt.savefig(plot_dir/"mag_vs_distance.png", dpi=140)
        plt.close()

def save_graph_plot(out_dir: str, G: nx.Graph, anomalies: set[int]):
    plot_dir = Path(out_dir)/"plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    # Use sky positions if available
    pos = {}
    for n in G.nodes():
        ra = G.nodes[n].get("ra")
        dec = G.nodes[n].get("dec")
        if ra is not None and dec is not None:
            pos[n] = (ra, dec)

    if not pos:
        pos = nx.spring_layout(G, seed=42)

    node_colors = ["red" if n in anomalies else "blue" for n in G.nodes()]
    plt.figure(figsize=(10,8))
    nx.draw(G, pos, node_color=node_colors, node_size=20, with_labels=False, alpha=0.8)
    plt.title("Graph (anomalies in red)")
    plt.tight_layout()
    plt.savefig(plot_dir/"graph_anomalies.png", dpi=140)
    plt.close()
