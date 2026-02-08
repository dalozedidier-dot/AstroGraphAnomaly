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
        is_anom = (df_scored["anomaly_label"].values == -1) if "anomaly_label" in df_scored.columns else None

        if is_anom is None:
            plt.scatter(df_scored["distance"], df_scored["phot_g_mean_mag"], s=6, alpha=0.7)
        else:
            plt.scatter(df_scored.loc[~is_anom, "distance"], df_scored.loc[~is_anom, "phot_g_mean_mag"], s=6, alpha=0.6)
            plt.scatter(df_scored.loc[is_anom, "distance"], df_scored.loc[is_anom, "phot_g_mean_mag"], s=18, alpha=0.9)

        plt.title("Magnitude vs distance")
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
    Stratégie:
      - base: spring_layout (donc pos complet)
      - overwrite: (ra,dec) si dispo, sinon node["pos"] si dispo
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if G.number_of_nodes() == 0:
        return

    # Base layout => garantit une position pour tous les nœuds
    pos = nx.spring_layout(G, seed=42)

    def _is_finite(x) -> bool:
        try:
            return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        except Exception:
            return False

    # Overwrite avec positions sky si disponibles
    for n in G.nodes():
        data = G.nodes[n]

        # 1) pos explicite
        p = data.get("pos", None)
        if isinstance(p, (tuple, list)) and len(p) == 2 and _is_finite(p[0]) and _is_finite(p[1]):
            pos[n] = (float(p[0]), float(p[1]))
            continue

        # 2) ra/dec
        ra = data.get("ra", None)
        dec = data.get("dec", None)
        if _is_finite(ra) and _is_finite(dec):
            pos[n] = (float(ra), float(dec))

    # anomalies: tolérer int/str mismatch
    anomalies_str = set(str(x) for x in anomalies)
    node_colors = ["red" if (n in anomalies or str(n) in anomalies_str) else "blue" for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=20, with_labels=False, alpha=0.85, width=0.4)
    plt.title("Graph (anomalies en rouge)")
    plt.tight_layout()
    plt.savefig(plot_dir / "graph_anomalies.png", dpi=160)
    plt.close()
