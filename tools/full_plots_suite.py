#!/usr/bin/env python3
"""
tools/full_plots_suite.py

Generate the "full plots" gallery for a run (offline-friendly).

This script is used by CI like:
  python tools/full_plots_suite.py \
    --scored results/<run>/scored.csv \
    --out results/<run>/plots \
    --top-k 30 \
    --graph results/<run>/graph_full.graphml \
    --write-enriched

Outputs (into --out):
  - score_hist.png
  - score_rank_curve.png
  - score_ecdf.png
  - ra_dec_score.png
  - pca_2d.png
  - mean_features_anom_vs_normal.png
  - top_anomalies_scores.png
  - mag_vs_distance.png
  - graph_communities_anomalies.png   (if --graph provided)

If --write-enriched is set, writes:
  - <scored_dir>/scored_enriched.csv

Matplotlib >= 3.9 requires an explicit Axes for colorbar() in some cases.
We always pass ax=... to avoid:
  ValueError: Unable to determine Axes to steal space for Colorbar
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Force non-interactive backend (CI / headless)
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _minmax01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    denom = mx - mn
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(x, dtype=float)
    return (x - mn) / denom


def _pick(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_score_col(df: pd.DataFrame) -> str | None:
    return _pick(df, ["anomaly_score", "score"])


def plot_score_hist(df: pd.DataFrame, out_dir: Path) -> None:
    col = _get_score_col(df)
    if not col:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(df[col].values, bins=60)
    plt.title("Distribution of anomaly scores")
    plt.xlabel(col)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_dir / "score_hist.png", dpi=200)
    plt.close()


def plot_score_rank_curve(df: pd.DataFrame, out_dir: Path) -> None:
    """Always-producible plot: score vs rank (descending)."""
    col = _get_score_col(df)
    if not col:
        return
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if s.size == 0:
        return
    s = np.sort(s)[::-1]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(np.arange(1, s.size + 1), s)
    ax.set_title("Anomaly score rank curve")
    ax.set_xlabel("rank (1 = highest score)")
    ax.set_ylabel(col)
    fig.tight_layout()
    fig.savefig(out_dir / "score_rank_curve.png", dpi=200)
    plt.close(fig)


def plot_score_ecdf(df: pd.DataFrame, out_dir: Path) -> None:
    """Always-producible plot: empirical CDF of scores."""
    col = _get_score_col(df)
    if not col:
        return
    s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().values
    if s.size == 0:
        return
    s = np.sort(s)
    y = np.arange(1, s.size + 1) / float(s.size)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(s, y)
    ax.set_title("Anomaly score ECDF")
    ax.set_xlabel(col)
    ax.set_ylabel("ECDF")  # 0..1
    ax.set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(out_dir / "score_ecdf.png", dpi=200)
    plt.close(fig)


def plot_ra_dec_score(df: pd.DataFrame, out_dir: Path) -> None:
    if not {"ra", "dec"}.issubset(df.columns):
        return
    score_col = _pick(df, ["anomaly_score_norm", "anomaly_score", "score"])
    if not score_col:
        return
    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(df["ra"].values, df["dec"].values, c=df[score_col].values, s=10)
    ax.set_title("RA/DEC colored by anomaly score")
    ax.set_xlabel("ra (deg)")
    ax.set_ylabel("dec (deg)")
    fig.colorbar(sc, ax=ax, label=score_col)
    fig.tight_layout()
    fig.savefig(out_dir / "ra_dec_score.png", dpi=200)
    plt.close(fig)


def plot_pca_2d(df: pd.DataFrame, out_dir: Path) -> None:
    # Optional dependency; skip cleanly if unavailable.
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return

    feature_candidates = [
        "parallax",
        "pmra",
        "pmdec",
        "phot_g_mean_mag",
        "distance",
    ]
    feats = [c for c in feature_candidates if c in df.columns]
    if len(feats) < 2:
        return

    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(df[feats].median(numeric_only=True)).values
    Xs = StandardScaler().fit_transform(X)
    p = PCA(n_components=2, random_state=0).fit_transform(Xs)

    score_col = _pick(df, ["anomaly_score_norm", "anomaly_score", "score"])
    c = df[score_col].values if score_col else None

    fig, ax = plt.subplots(figsize=(9, 7))
    sc = ax.scatter(p[:, 0], p[:, 1], c=c, s=10) if c is not None else ax.scatter(p[:, 0], p[:, 1], s=10)
    ax.set_title("PCA 2D projection")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    if c is not None:
        fig.colorbar(sc, ax=ax, label=score_col)
    fig.tight_layout()
    fig.savefig(out_dir / "pca_2d.png", dpi=200)
    plt.close(fig)


def plot_mean_features(df: pd.DataFrame, out_dir: Path) -> None:
    if "anomaly_label" not in df.columns:
        return

    numeric_cols = [
        c for c in df.columns
        if c not in {"source_id"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Avoid plotting score columns twice; keep a compact set.
    keep = [c for c in numeric_cols if c not in {"anomaly_score_norm"}]
    keep = keep[:8] if len(keep) > 8 else keep
    if not keep:
        return

    means = df.groupby("anomaly_label")[keep].mean(numeric_only=True)
    means = means.reindex(sorted(means.index.tolist()))

    fig, ax = plt.subplots(figsize=(12, 6))
    means.T.plot(kind="bar", ax=ax)
    ax.set_title("Mean features: anomalies vs normal")
    ax.set_xlabel("feature")
    ax.set_ylabel("mean value")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "mean_features_anom_vs_normal.png", dpi=200)
    plt.close(fig)


def plot_top_anomalies(df: pd.DataFrame, out_dir: Path, top_k: int) -> None:
    score_col = _get_score_col(df)
    if not score_col:
        return

    top = df.sort_values(score_col, ascending=False).head(top_k).copy()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(top)), top[score_col].values)
    ax.set_title(f"Top {top_k} anomalies by score")
    ax.set_xlabel("rank")
    ax.set_ylabel(score_col)
    fig.tight_layout()
    fig.savefig(out_dir / "top_anomalies_scores.png", dpi=200)
    plt.close(fig)


def plot_mag_vs_distance(df: pd.DataFrame, out_dir: Path) -> None:
    mag_col = _pick(df, ["phot_g_mean_mag", "mag", "phot_g_mean_mag"])
    dist_col = _pick(df, ["distance"])
    if not mag_col or not dist_col:
        return
    score_col = _pick(df, ["anomaly_score_norm", "anomaly_score", "score"])

    fig, ax = plt.subplots(figsize=(9, 7))
    if score_col:
        sc = ax.scatter(df[dist_col].values, df[mag_col].values, c=df[score_col].values, s=10)
        fig.colorbar(sc, ax=ax, label=score_col)
    else:
        ax.scatter(df[dist_col].values, df[mag_col].values, s=10)

    ax.set_title("Magnitude vs distance")
    ax.set_xlabel(dist_col)
    ax.set_ylabel(mag_col)
    # In astronomy, smaller magnitude is brighter; invert y for intuition.
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "mag_vs_distance.png", dpi=200)
    plt.close(fig)


def _compute_graph_enrichment(df: pd.DataFrame, graph_path: Path) -> pd.DataFrame:
    import networkx as nx
    from networkx.algorithms import community as nx_comm

    G = nx.read_graphml(graph_path)

    degree = dict(G.degree())
    clustering = nx.clustering(G)
    kcore = nx.core_number(G)
    betweenness = nx.betweenness_centrality(G, normalized=True)

    # communities: deterministic order by (size desc, min node id)
    comms = list(nx_comm.greedy_modularity_communities(G))

    def _comm_key(cset) -> tuple[int, str]:
        mins = min(cset) if cset else ""
        return (-len(cset), str(mins))

    comms_sorted = sorted(comms, key=_comm_key)

    community_id: dict[str, int] = {}
    for i, cset in enumerate(comms_sorted):
        for n in cset:
            community_id[str(n)] = i

    sid = df["source_id"].astype(str)
    df2 = df.copy()

    score_col = _get_score_col(df2)
    if score_col and "anomaly_score_norm" not in df2.columns:
        df2["anomaly_score_norm"] = _minmax01(df2[score_col].values)

    df2["degree"] = sid.map(lambda s: int(degree.get(s, 0)))
    df2["clustering"] = sid.map(lambda s: float(clustering.get(s, 0.0)))
    df2["kcore"] = sid.map(lambda s: int(kcore.get(s, 0)))
    df2["betweenness"] = sid.map(lambda s: float(betweenness.get(s, 0.0)))
    df2["community_id"] = sid.map(lambda s: int(community_id.get(s, -1)))

    return df2


def plot_graph_communities(df_enriched: pd.DataFrame, graph_path: Path, out_dir: Path) -> None:
    import networkx as nx

    if not {"ra", "dec", "community_id"}.issubset(df_enriched.columns):
        return

    G = nx.read_graphml(graph_path)

    # Positions: use RA/DEC when present; fallback to spring layout if missing.
    pos: dict[str, tuple[float, float]] = {}
    have_ra = True
    for n, d in G.nodes(data=True):
        if "ra" in d and "dec" in d:
            pos[str(n)] = (float(d["ra"]), float(d["dec"]))
        else:
            have_ra = False
            break
    if not have_ra:
        pos = nx.spring_layout(G, seed=0)

    sid = df_enriched["source_id"].astype(str).values
    comm = df_enriched["community_id"].values
    is_anom = (df_enriched.get("anomaly_label", 1).values == -1)

    xs = np.array([pos.get(s, (np.nan, np.nan))[0] for s in sid], dtype=float)
    ys = np.array([pos.get(s, (np.nan, np.nan))[1] for s in sid], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 8))

    # Draw edges lightly
    try:
        for u, v in G.edges():
            pu = pos.get(str(u))
            pv = pos.get(str(v))
            if pu is None or pv is None:
                continue
            ax.plot([pu[0], pv[0]], [pu[1], pv[1]], linewidth=0.3, alpha=0.08)
    except Exception:
        pass

    sizes = np.where(is_anom, 30.0, 10.0)
    sc = ax.scatter(xs, ys, c=comm, s=sizes, cmap="tab20", alpha=0.95)

    ax.set_title("Graph communities (nodes colored) + anomalies (larger)")
    ax.set_xlabel("ra (deg)" if have_ra else "x")
    ax.set_ylabel("dec (deg)" if have_ra else "y")

    # Critical fix: always provide ax for colorbar (Matplotlib >= 3.9)
    fig.colorbar(sc, ax=ax, label="Community ID")

    fig.tight_layout()
    fig.savefig(out_dir / "graph_communities_anomalies.png", dpi=200)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="Path to scored.csv")
    ap.add_argument("--out", required=True, help="Directory to write plots into")
    ap.add_argument("--top-k", type=int, default=30, help="Top-k for top anomalies plot")
    ap.add_argument("--graph", default=None, help="Path to graph_full.graphml (optional)")
    ap.add_argument("--write-enriched", action="store_true", help="Write scored_enriched.csv next to scored.csv")
    args = ap.parse_args()

    scored_path = Path(args.scored)
    out_dir = Path(args.out)

    _safe_mkdir(out_dir)

    df = pd.read_csv(scored_path)

    score_col = _get_score_col(df)
    if score_col and "anomaly_score_norm" not in df.columns:
        df["anomaly_score_norm"] = _minmax01(df[score_col].values)

    # Always-on plots
    plot_score_hist(df, out_dir)
    plot_score_rank_curve(df, out_dir)
    plot_score_ecdf(df, out_dir)
    plot_ra_dec_score(df, out_dir)
    plot_pca_2d(df, out_dir)
    plot_mean_features(df, out_dir)
    plot_top_anomalies(df, out_dir, top_k=args.top_k)
    plot_mag_vs_distance(df, out_dir)

    # Graph enrichment + plot
    if args.graph:
        graph_path = Path(args.graph)
        if graph_path.exists():
            df_enriched = _compute_graph_enrichment(df, graph_path)
            if args.write_enriched:
                enriched_path = scored_path.with_name("scored_enriched.csv")
                df_enriched.to_csv(enriched_path, index=False)
            plot_graph_communities(df_enriched, graph_path, out_dir)
        else:
            if args.write_enriched:
                enriched_path = scored_path.with_name("scored_enriched.csv")
                df.to_csv(enriched_path, index=False)
    else:
        if args.write_enriched:
            enriched_path = scored_path.with_name("scored_enriched.csv")
            df.to_csv(enriched_path, index=False)

    print(f"[full_plots_suite] wrote plots into: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
