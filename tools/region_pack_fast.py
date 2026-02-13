#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/region_pack_fast.py

Fast region pack scoring for Gaia DR3 "derived tables" such as:
- galaxy_candidates (GalaxyCandidates_*.csv.gz)
- vari_summary      (VariSummary_*.csv.gz)
- (optional) galaxy_catalogue_name mapping (GalaxyCatalogueName_*.csv.gz)

Goal
- Score anomalies per region file quickly (tabular-only, no ra/dec required)
- Produce lightweight artifacts per region:
  - scored.csv.gz
  - 01_embedding_pca.png
  - summary.json
  - index.html
- Produce an aggregated file:
  - <kind>_scored_all.csv.gz

Important
- Input files are Astropy ECSV exported as .csv.gz (with many leading '# ...' lines).
  We parse them with pandas using comment='#'.

This tool is intentionally separate from the core sky-graph pipeline (run_workflow.py),
because these DR3 derived tables typically do not include ra/dec.
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")  # CI-friendly
import matplotlib.pyplot as plt


RID_RE = re.compile(r"_(\d{6})-(\d{6})\.csv\.gz$", re.IGNORECASE)


def _read_ecsv_csv_gz(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read Astropy ECSV stored as .csv.gz into a DataFrame."""
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        df = pd.read_csv(f, comment="#", nrows=nrows)
    return df


def _infer_region_id(path: Path) -> str:
    m = RID_RE.search(path.name)
    if not m:
        # fallback: best-effort
        stem = path.name.replace(".csv.gz", "")
        return stem.split("_")[-1]
    return f"{m.group(1)}-{m.group(2)}"


def _robust_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), x, np.nan)
    finite = np.isfinite(x)
    if not np.any(finite):
        return np.zeros_like(x, dtype=float)
    lo = float(np.nanpercentile(x, 5))
    hi = float(np.nanpercentile(x, 95))
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        lo = float(np.nanmin(x))
        hi = float(np.nanmax(x))
    if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) < 1e-12:
        return np.zeros_like(x, dtype=float)
    y = (x - lo) / (hi - lo)
    y = np.clip(y, 0.0, 1.0)
    y = np.where(np.isfinite(y), y, 0.0)
    return y


def _select_numeric_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Return (X_df, feature_names). Keeps numeric columns excluding ids."""
    num = df.select_dtypes(include=[np.number]).copy()
    for col in ["solution_id", "source_id"]:
        if col in num.columns:
            num = num.drop(columns=[col])

    # remove all-NaN columns
    nunique = num.nunique(dropna=True)
    keep_cols = [c for c in num.columns if nunique.get(c, 0) > 1]
    num = num[keep_cols]

    # impute NaNs with median
    for c in num.columns:
        med = float(num[c].median(skipna=True)) if np.isfinite(num[c].median(skipna=True)) else 0.0
        num[c] = num[c].fillna(med)

    return num, list(num.columns)


def _score_robust_zscore(X: np.ndarray) -> np.ndarray:
    """Robust z-score anomaly score: higher = more anomalous."""
    X = np.asarray(X, dtype=float)
    med = np.nanmedian(X, axis=0)
    mad = np.nanmedian(np.abs(X - med), axis=0)
    mad = np.where(mad < 1e-12, 1e-12, mad)
    z = 0.6745 * (X - med) / mad
    score = np.nanmedian(np.abs(z), axis=1)
    score = np.where(np.isfinite(score), score, 0.0)
    return score


def _score_isolation_forest(X_scaled: np.ndarray, seed: int, contamination: float) -> np.ndarray:
    model = IsolationForest(
        n_estimators=300,
        contamination=float(contamination),
        random_state=int(seed),
        n_jobs=-1,
    )
    model.fit(X_scaled)
    # decision_function: higher = more normal; negate to get anomaly score
    scores = -model.decision_function(X_scaled)
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores


def _top_k_labels(scores: np.ndarray, k: int) -> np.ndarray:
    k = int(max(1, k))
    idx = np.argsort(-scores)  # descending
    labels = np.ones(scores.shape[0], dtype=int)
    labels[idx[:k]] = -1
    return labels


def _make_index_html(
    out_dir: Path,
    region_id: str,
    kind: str,
    engine: str,
    top_df: pd.DataFrame,
    summary: Dict[str, Any],
) -> None:
    rows = []
    for _, r in top_df.iterrows():
        sid = html.escape(str(r.get("source_id", "")))
        sc = html.escape(f"{float(r.get('anomaly_score', 0.0)):.6f}")
        extra = []
        for c in ["vari_best_class_name", "vari_best_class_score", "classlabel_dsc", "classlabel_dsc_joint", "classlabel_oa"]:
            if c in r.index and pd.notna(r[c]):
                extra.append(f"{c}={html.escape(str(r[c]))}")
        extra_s = html.escape(", ".join(extra)) if extra else ""
        rows.append(f"<tr><td><code>{sid}</code></td><td>{sc}</td><td>{extra_s}</td></tr>")

    html_doc = f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8" />
  <title>AstroGraphAnomaly Region Pack {html.escape(kind)} {html.escape(region_id)}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; max-width: 1100px; }}
    img {{ max-width: 100%; border-radius: 10px; box-shadow: 0 6px 18px rgba(0,0,0,.12); }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e6e6e6; padding: 8px 10px; text-align: left; font-size: 14px; }}
    th {{ background: #fafafa; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }}
    .meta {{ color: #444; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Region Pack {html.escape(kind)} {html.escape(region_id)}</h1>
  <p class="meta">Engine: <b>{html.escape(engine)}</b> · n_rows: <b>{int(summary.get("n_rows", 0))}</b> · n_features: <b>{int(summary.get("n_features", 0))}</b></p>

  <div class="grid">
    <div>
      <h2>Embedding PCA</h2>
      <img src="01_embedding_pca.png" alt="Embedding PCA" />
    </div>

    <div>
      <h2>Top anomalies</h2>
      <table>
        <thead><tr><th>source_id</th><th>anomaly_score</th><th>notes</th></tr></thead>
        <tbody>
          {''.join(rows)}
        </tbody>
      </table>
    </div>

    <div>
      <h2>Summary</h2>
      <pre>{html.escape(json.dumps(summary, indent=2, ensure_ascii=False))}</pre>
    </div>
  </div>
</body>
</html>
"""
    (out_dir / "index.html").write_text(html_doc, encoding="utf-8")


def _plot_pca(
    out_png: Path,
    emb2: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    title: str,
) -> None:
    s01 = _robust_01(scores)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(emb2[:, 0], emb2[:, 1], c=s01, s=8, alpha=0.85)
    # highlight anomalies
    mask = labels == -1
    if np.any(mask):
        ax.scatter(emb2[mask, 0], emb2[mask, 1], s=22, facecolors="none", edgecolors="black", linewidths=0.8)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def run_one_region(
    kind: str,
    in_path: Path,
    out_root: Path,
    engine: str,
    top_k: int,
    max_rows: Optional[int],
    seed: int,
    contamination: float,
    catalogue_maps: Optional[List[Path]] = None,
) -> Path:
    region_id = _infer_region_id(in_path)
    out_dir = out_root / f"region_{region_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _read_ecsv_csv_gz(in_path, nrows=max_rows)
    if "source_id" not in df.columns:
        raise ValueError(f"{in_path.name}: missing required column 'source_id'")

    df = df.copy()
    df["source_id"] = df["source_id"].astype("int64", errors="ignore")

    # Optional catalogue maps: add n_catalogues per source_id
    if catalogue_maps:
        maps = []
        for mp in catalogue_maps:
            mdf = _read_ecsv_csv_gz(mp, nrows=None)
            if "source_id" in mdf.columns and "catalogue_id" in mdf.columns:
                maps.append(mdf[["source_id", "catalogue_id"]])
        if maps:
            allmap = pd.concat(maps, ignore_index=True)
            allmap["source_id"] = allmap["source_id"].astype("int64", errors="ignore")
            counts = allmap.groupby("source_id")["catalogue_id"].nunique().rename("n_catalogues").reset_index()
            df = df.merge(counts, on="source_id", how="left")
            df["n_catalogues"] = df["n_catalogues"].fillna(0).astype(int)

    X_df, feature_names = _select_numeric_features(df)
    if X_df.shape[1] < 3:
        raise ValueError(f"{in_path.name}: not enough numeric features after cleaning ({X_df.shape[1]})")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.to_numpy(dtype=float))

    if engine == "robust_zscore":
        scores = _score_robust_zscore(X_scaled)
    elif engine == "isolation_forest":
        scores = _score_isolation_forest(X_scaled, seed=seed, contamination=contamination)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    labels = _top_k_labels(scores, k=top_k)

    scored = df[["source_id"]].copy()
    scored["anomaly_score"] = scores
    scored["anomaly_label"] = labels
    scored["region_id"] = region_id

    # carry a few useful fields if present
    carry_cols = []
    for c in [
        "vari_best_class_name",
        "vari_best_class_score",
        "classlabel_dsc",
        "classlabel_dsc_joint",
        "classlabel_oa",
        "redshift_ugc",
        "n_transits",
        "n_catalogues",
    ]:
        if c in df.columns:
            carry_cols.append(c)
    for c in carry_cols:
        scored[c] = df[c].values

    scored_path = out_dir / "scored.csv.gz"
    scored.to_csv(scored_path, index=False, compression="gzip")

    # PCA embedding
    pca = PCA(n_components=2, random_state=int(seed))
    emb2 = pca.fit_transform(X_scaled)
    _plot_pca(
        out_png=out_dir / "01_embedding_pca.png",
        emb2=emb2,
        scores=scores,
        labels=labels,
        title=f"{kind} {region_id} · PCA embedding (n={len(df)})",
    )

    # Summary
    summary = {
        "kind": kind,
        "region_id": region_id,
        "engine": engine,
        "top_k": int(top_k),
        "n_rows": int(df.shape[0]),
        "n_features": int(X_df.shape[1]),
        "features_sample": feature_names[:25],
        "score_min": float(np.min(scores)) if len(scores) else 0.0,
        "score_p50": float(np.percentile(scores, 50)) if len(scores) else 0.0,
        "score_p95": float(np.percentile(scores, 95)) if len(scores) else 0.0,
        "score_max": float(np.max(scores)) if len(scores) else 0.0,
        "max_rows_limit": int(max_rows) if max_rows is not None else None,
        "input_file": in_path.name,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Index HTML
    top_df = scored.sort_values("anomaly_score", ascending=False).head(int(top_k))
    _make_index_html(out_dir=out_dir, region_id=region_id, kind=kind, engine=engine, top_df=top_df, summary=summary)

    return out_dir


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fast region pack scoring for Gaia DR3 derived tables.")
    ap.add_argument("--kind", choices=["galaxy_candidates", "vari_summary"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more input glob patterns or file paths.")
    ap.add_argument("--out", required=True, help="Output directory root (e.g., results/region_pack_fast/galaxy_candidates)")
    ap.add_argument("--engine", choices=["robust_zscore", "isolation_forest"], default="robust_zscore")
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--max-rows", type=int, default=0, help="0 disables row limit (use full file).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--contamination", type=float, default=0.02, help="Only used for isolation_forest.")
    ap.add_argument("--catalogue-maps", nargs="*", default=None, help="Optional GalaxyCatalogueName_*.csv.gz files to enrich n_catalogues.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # resolve inputs globs
    in_paths: List[Path] = []
    for item in args.inputs:
        p = Path(item)
        if "*" in item or "?" in item or ("[" in item and "]" in item):
            in_paths.extend(sorted(Path().glob(item)))
        elif p.exists():
            in_paths.append(p)
        else:
            # relative to repo root if present
            if Path(item).exists():
                in_paths.append(Path(item))
    in_paths = [p for p in in_paths if p.exists()]

    if not in_paths:
        raise SystemExit("No input files resolved from --inputs")

    catalogue_maps = [Path(p) for p in (args.catalogue_maps or []) if Path(p).exists()]
    max_rows = int(args.max_rows) if int(args.max_rows) > 0 else None

    all_scored: List[pd.DataFrame] = []
    for ip in in_paths:
        out_dir = run_one_region(
            kind=str(args.kind),
            in_path=ip,
            out_root=out_root,
            engine=str(args.engine),
            top_k=int(args.top_k),
            max_rows=max_rows,
            seed=int(args.seed),
            contamination=float(args.contamination),
            catalogue_maps=catalogue_maps if catalogue_maps else None,
        )
        # collect scored for aggregate
        sc = pd.read_csv(out_dir / "scored.csv.gz", compression="gzip")
        all_scored.append(sc)

    agg = pd.concat(all_scored, ignore_index=True) if all_scored else pd.DataFrame()
    agg_path = out_root / f"{args.kind}_scored_all.csv.gz"
    agg.to_csv(agg_path, index=False, compression="gzip")

    print(f"[region_pack_fast] wrote: {agg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
