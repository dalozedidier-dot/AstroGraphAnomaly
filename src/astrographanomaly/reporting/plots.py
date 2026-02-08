from __future__ import annotations

from pathlib import Path
import math
from typing import Iterable, Optional, Tuple, Dict, Any

import matplotlib.pyplot as plt
import networkx as nx


# -------------------------------
# Helpers
# -------------------------------

def _is_finite(x) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
    except Exception:
        return False


def _infer_score_direction(df_scored) -> bool:
    """Return True if higher anomaly_score seems more anomalous, else False.

    Heuristic: compare median score for label=-1 vs label=+1.
    If labels missing, default True (current project convention).
    """
    if "anomaly_score" not in df_scored.columns:
        return True
    if "anomaly_label" not in df_scored.columns:
        return True

    an = df_scored[df_scored["anomaly_label"] == -1]["anomaly_score"]
    no = df_scored[df_scored["anomaly_label"] != -1]["anomaly_score"]
    if len(an) == 0 or len(no) == 0:
        return True

    return float(an.median()) >= float(no.median())


def _topk_ids(df_scored, k: int = 30) -> Tuple[list, bool]:
    """Return list of top-k source_id according to anomaly_score direction + direction flag."""
    higher_more_anom = _infer_score_direction(df_scored)
    if "anomaly_score" not in df_scored.columns or "source_id" not in df_scored.columns:
        return [], higher_more_anom

    df = df_scored[["source_id", "anomaly_score"]].dropna()
    if higher_more_anom:
        df = df.sort_values("anomaly_score", ascending=False)
    else:
        df = df.sort_values("anomaly_score", ascending=True)

    return df["source_id"].head(int(k)).tolist(), higher_more_anom


def _safe_numeric_series(df, col: str):
    if col not in df.columns:
        return None
    s = df[col].replace([math.inf, -math.inf], math.nan)
    return s


# -------------------------------
# Public API used by pipeline
# -------------------------------

def save_basic_plots(out_dir: str | Path, df_scored) -> None:
    """Curated set of plots: pertinent + lisible + reproductible.

    Exports into: <out_dir>/plots/*.png

    Always (if columns present):
      - score_hist.png
      - ra_dec_score.png
      - mag_vs_distance.png
      - top_anomalies_scores.png
      - mean_features_anom_vs_normal.png
      - pca_2d.png (numeric columns)
      - cmd_bp_rp_vs_g.png (Gaia only if bp_rp exists)
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    higher_more_anom = _infer_score_direction(df_scored)

    # --- 1) Score distribution
    if "anomaly_score" in df_scored.columns:
        plt.figure(figsize=(9, 5.5))
        s = _safe_numeric_series(df_scored, "anomaly_score").dropna().to_numpy(float)
        plt.hist(s, bins=60, alpha=0.85)
        plt.title("Distribution des scores d'anomalie")
        plt.xlabel("Score d'anomalie" + (" (plus élevé = plus anomal)" if higher_more_anom else " (plus faible = plus anomal)"))
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.savefig(plot_dir / "score_hist.png", dpi=180)
        plt.close()

    # --- 2) RA/Dec colored by score (spatial diagnostic)
    if {"ra", "dec", "anomaly_score"}.issubset(df_scored.columns):
        plt.figure(figsize=(10, 7))
        plt.scatter(df_scored["ra"], df_scored["dec"], c=df_scored["anomaly_score"], s=22, alpha=0.85)
        plt.colorbar(label="Score d'anomalie")
        plt.title("Distribution spatiale (RA vs Dec) colorée par score")
        plt.xlabel("Ascension Droite (RA)")
        plt.ylabel("Déclinaison (Dec)")
        plt.tight_layout()
        plt.savefig(plot_dir / "ra_dec_score.png", dpi=180)
        plt.close()

    # --- 3) Magnitude vs distance (structure astro + anomalies)
    if "phot_g_mean_mag" in df_scored.columns and "distance" in df_scored.columns:
        plt.figure(figsize=(9, 5.5))
        if "anomaly_label" in df_scored.columns:
            is_anom = df_scored["anomaly_label"].values == -1
            plt.scatter(df_scored.loc[~is_anom, "distance"], df_scored.loc[~is_anom, "phot_g_mean_mag"], s=8, alpha=0.45)
            plt.scatter(df_scored.loc[is_anom, "distance"], df_scored.loc[is_anom, "phot_g_mean_mag"], s=28, alpha=0.90)
  # anomalies
        else:
            plt.scatter(df_scored["distance"], df_scored["phot_g_mean_mag"], s=8, alpha=0.65)

        plt.title("Magnitude vs distance (anomalies surlignées si disponibles)")
        plt.xlabel("Distance (pc)")
        plt.ylabel("phot_g_mean_mag")
        plt.tight_layout()
        plt.savefig(plot_dir / "mag_vs_distance.png", dpi=180)
        plt.close()

    # --- 4) Gaia CMD (BP-RP vs G) if available (aesthetically strong + informative)
    if "bp_rp" in df_scored.columns and "phot_g_mean_mag" in df_scored.columns:
        plt.figure(figsize=(10, 7))
        if "anomaly_label" in df_scored.columns:
            is_anom = df_scored["anomaly_label"].values == -1
            plt.scatter(df_scored.loc[~is_anom, "bp_rp"], df_scored.loc[~is_anom, "phot_g_mean_mag"], s=6, alpha=0.45)
            plt.scatter(df_scored.loc[is_anom, "bp_rp"], df_scored.loc[is_anom, "phot_g_mean_mag"], s=18, alpha=0.90)
        else:
            plt.scatter(df_scored["bp_rp"], df_scored["phot_g_mean_mag"], s=6, alpha=0.65)

        # CMD convention: brighter is higher up -> invert magnitude axis
        plt.gca().invert_yaxis()
        plt.title("Diagramme Couleur–Magnitude Gaia (BP-RP vs G)")
        plt.xlabel("BP - RP [mag]")
        plt.ylabel("G [mag]")
        plt.tight_layout()
        plt.savefig(plot_dir / "cmd_bp_rp_vs_g.png", dpi=180)
        plt.close()

    # --- 5) Mean feature comparison (anomalous vs normal) — concise and interpretable
    if "anomaly_label" in df_scored.columns:
        cols_pref = [
            "phot_g_mean_mag",
            "bp_rp",
            "parallax",
            "pmra",
            "pmdec",
            "distance",
        ]
        cols = [c for c in cols_pref if c in df_scored.columns]
        if len(cols) >= 2:
            an = df_scored[df_scored["anomaly_label"] == -1]
            no = df_scored[df_scored["anomaly_label"] != -1]

            # means, ignoring nan/inf
            def _mean(d, c):
                s = _safe_numeric_series(d, c)
                return float(s.dropna().to_numpy(float).mean()) if s is not None and s.dropna().shape[0] else float("nan")

            an_means = [_mean(an, c) for c in cols]
            no_means = [_mean(no, c) for c in cols]

            x = range(len(cols))
            width = 0.38
            plt.figure(figsize=(12, 6))
            plt.bar([i - width / 2 for i in x], an_means, width=width, label="Anomalous")
            plt.bar([i + width / 2 for i in x], no_means, width=width, label="Normal")
            plt.xticks(list(x), cols, rotation=0)
            plt.title("Comparaison des moyennes de features : anomalous vs normal")
            plt.ylabel("Valeur moyenne")
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / "mean_features_anom_vs_normal.png", dpi=180)
            plt.close()

    # --- 6) Top anomalies bar (most communicable “ranked list” plot)
    if {"source_id", "anomaly_score"}.issubset(df_scored.columns):
        top_ids, higher_more_anom = _topk_ids(df_scored, k=30)
        if top_ids:
            top_df = df_scored[df_scored["source_id"].isin(top_ids)][["source_id", "anomaly_score"]].dropna()
            # preserve order
            order = {sid: i for i, sid in enumerate(top_ids)}
            top_df["__ord"] = top_df["source_id"].map(order)
            top_df = top_df.sort_values("__ord")

            plt.figure(figsize=(14, 5.5))
            plt.bar(range(len(top_df)), top_df["anomaly_score"].to_numpy(float), alpha=0.9)
            plt.xticks(range(len(top_df)), top_df["source_id"].astype(str).tolist(), rotation=90)
            plt.title("Top anomalies : score par source_id")
            plt.xlabel("source_id (candidats anomalies)")
            plt.ylabel("Score d'anomalie" + (" (haut = anomal)" if higher_more_anom else " (bas = anomal)"))
            plt.tight_layout()
            plt.savefig(plot_dir / "top_anomalies_scores.png", dpi=180)
            plt.close()

    # --- 7) PCA 2D (numeric columns) — compact global structure diagnostic
    try:
        from sklearn.decomposition import PCA
        import numpy as np
        import pandas as pd
    except Exception:
        PCA = None  # type: ignore

    if PCA is not None:
        # choose numeric cols excluding obvious ids/labels
        import numpy as np
        import pandas as pd

        numeric_cols = [c for c in df_scored.columns if c not in ("source_id", "anomaly_label") and pd.api.types.is_numeric_dtype(df_scored[c])]
        if len(numeric_cols) >= 3:
            X = df_scored[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
            if X.shape[0] >= 10:
                pca = PCA(n_components=2, random_state=42)
                Z = pca.fit_transform(X)

                plt.figure(figsize=(9.5, 6.5))
                if "anomaly_label" in df_scored.columns:
                    is_anom = df_scored["anomaly_label"].to_numpy(int) == -1
                    plt.scatter(Z[~is_anom, 0], Z[~is_anom, 1], s=10, alpha=0.5)
                    plt.scatter(Z[is_anom, 0], Z[is_anom, 1], s=28, alpha=0.9)
                elif "anomaly_score" in df_scored.columns:
                    plt.scatter(Z[:, 0], Z[:, 1], c=df_scored["anomaly_score"], s=12, alpha=0.85)
                    plt.colorbar(label="Score d'anomalie")
                else:
                    plt.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.75)

                evr = getattr(pca, "explained_variance_ratio_", None)
                subtitle = ""
                if evr is not None and len(evr) == 2:
                    subtitle = f" (EVR: {evr[0]:.2f}, {evr[1]:.2f})"
                plt.title("Projection PCA(2) sur colonnes numériques" + subtitle)
                plt.xlabel("PC1")
                plt.ylabel("PC2")
                plt.tight_layout()
                plt.savefig(plot_dir / "pca_2d.png", dpi=180)
                plt.close()


def save_graph_plot(out_dir: str | Path, G: nx.Graph, anomalies: set) -> None:
    """Graph plot: community-colored, anomalies emphasized.

    - positions: spring_layout (complete), overwritten by (ra,dec) or pos if present
    - communities: Louvain if available, else greedy modularity
    - anomalies: larger nodes
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    if G.number_of_nodes() == 0:
        return

    # base positions => complete coverage
    pos = nx.spring_layout(G, seed=42)

    # overwrite with sky coords if available
    for n in G.nodes():
        data = G.nodes[n]

        p = data.get("pos", None)
        if isinstance(p, (tuple, list)) and len(p) == 2 and _is_finite(p[0]) and _is_finite(p[1]):
            pos[n] = (float(p[0]), float(p[1]))
            continue

        ra = data.get("ra", None)
        dec = data.get("dec", None)
        if _is_finite(ra) and _is_finite(dec):
            pos[n] = (float(ra), float(dec))

    # communities
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, seed=42)
    except Exception:
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(G)
        except Exception:
            comms = []

    comm_id: Dict[Any, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            comm_id[n] = i

    anomalies_str = set(str(x) for x in anomalies)

    node_list = list(G.nodes())
    node_colors = [comm_id.get(n, 0) for n in node_list]
    node_sizes = [70 if (n in anomalies or str(n) in anomalies_str) else 22 for n in node_list]

    plt.figure(figsize=(10.5, 8.5))
    nx.draw(
        G,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        with_labels=False,
        alpha=0.88,
        width=0.4,
    )
    plt.title("Graphe (communautés) + anomalies (taille)")
    plt.tight_layout()
    plt.savefig(plot_dir / "graph_communities_anomalies.png", dpi=180)
    plt.close()
