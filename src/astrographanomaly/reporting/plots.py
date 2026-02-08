from __future__ import annotations

from pathlib import Path
import math
import matplotlib.pyplot as plt
import networkx as nx


def save_basic_plots(out_dir: str | Path, df_scored) -> None:
    """
    Plots robustes (ne cassent pas si certaines colonnes manquent).
    Sortie : <out_dir>/plots/*.png
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1) Distribution anomaly_score
    if "anomaly_score" in df_scored.columns:
        plt.figure(figsize=(8, 5))
        df_scored["anomaly_score"].hist(bins=60)
        plt.title("Anomaly score distribution (higher = more anomalous)")
        plt.xlabel("anomaly_score")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(plot_dir / "score_hist.png", dpi=160)
        plt.close()

    # 2) Mag vs distance si dispo
    if "phot_g_mean_mag" in df_scored.columns and "distance" in df_scored.columns:
        plt.figure(figsize=(8, 5))
        is_anom = df_scored.get("anomaly_label", 1).values == -1
        plt.scatter(df_scored.loc[~is_anom, "distance"], df_scored.loc[~is_anom, "phot_g_mean_mag"], s=6, alpha=0.6)
        plt.scatter(df_scored.loc[is_anom, "distance"], df_scored.loc[is_anom, "phot_g_mean_mag"], s=18, alpha=0.9)
        plt.title("Magnitude vs distance (anomalies highlighted)")
        plt.xlabel("distance (pc)")
        plt.ylabel("phot_g_mean_mag")
        plt.tight_layout()
        plt.savefig(plot_dir / "mag_vs_distance.png", dpi=160)
        plt.close()

    # 3) RA/Dec coloré par score si dispo
    if {"ra", "dec", "anomaly_score"}.issubset(df_scored.columns):
        plt.figure(figsize=(10, 7))
        plt.scatter(df_scored["ra"], df_scored["dec"], c=df_scored["anomaly_score"], s=35, alpha=0.85)
        plt.colorbar(label="Anomaly score")
        plt.title("Spatial distribution (RA vs Dec) colored by score")
        plt.xlabel("Right Ascension (RA)")
        plt.ylabel("Declination (Dec)")
        plt.tight_layout()
        plt.savefig(plot_dir / "ra_dec_score.png", dpi=160)
        plt.close()


def save_graph_plot(out_dir: str | Path, G: nx.Graph, anomalies: set) -> None:
    """
    Plot du graphe en garantissant une position pour chaque nœud.
    - utilise node["pos"] si présent
    - sinon utilise (ra, dec) si présents
    - sinon fallback spring_layout pour les nœuds manquants
    - ajoute aussi les clés str(node) pour éviter mismatch int/str
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pos: dict = {}

    def _is_finite(x) -> bool:
        try:
