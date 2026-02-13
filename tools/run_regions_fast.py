#!/usr/bin/env python3
"""
Runner rapide: anomalies (kNN distance), PCA2 embedding, outputs par région (fichier chunk).

Usage:
  python tools/run_regions_fast.py --kind galaxy_candidates --inputs "data/GalaxyCandidates_*.csv.gz" --out results/galaxy_candidates_fast
  python tools/run_regions_fast.py --kind vari_summary     --inputs "data/VariSummary_*.csv.gz"     --out results/vari_summary_fast

Notes:
- galaxy_candidates et vari_summary ne contiennent pas ra/dec.
  Pour les sky maps, enrichir via gaiadr3.gaia_source (ADQL) sur source_id.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


def read_ecsv_gz(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip", comment="#")


def infer_feature_columns(df: pd.DataFrame, id_col: str = "source_id") -> list[str]:
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    drop = {id_col, "solution_id"}

    feats = [c for c in num_cols if c not in drop]
    for c in bool_cols:
        if c not in drop and c not in feats:
            feats.append(c)

    good: list[str] = []
    for c in feats:
        s = df[c]
        if s.isna().all():
            continue
        if pd.api.types.is_numeric_dtype(s):
            if s.nunique(dropna=True) <= 1:
                continue
        else:
            if s.nunique() <= 1:
                continue
        good.append(c)
    return good


def rank01(z: np.ndarray) -> np.ndarray:
    r = pd.Series(z).rank(method="average").to_numpy()
    return (r - 1) / (len(r) - 1) if len(r) > 1 else np.zeros_like(r)


def fit_transform_features(df: pd.DataFrame, feature_cols: list[str], pca_dim: int = 10):
    X = df[feature_cols].copy()
    for c in feature_cols:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0))
    X_imp = imputer.fit_transform(X)
    X_sc = scaler.fit_transform(X_imp)

    pca_dim = min(pca_dim, X_sc.shape[1], 10)
    X_p = PCA(n_components=pca_dim, random_state=0).fit_transform(X_sc)
    emb2 = PCA(n_components=2, random_state=0).fit_transform(X_sc)
    return X_p, emb2


def knn_score(X_p: np.ndarray, k: int = 15) -> np.ndarray:
    n = len(X_p)
    k = min(k, max(6, int(math.sqrt(n) // 2)))
    nn = NearestNeighbors(n_neighbors=min(k, n), algorithm="auto", metric="euclidean")
    nn.fit(X_p)
    d, _ = nn.kneighbors(X_p, return_distance=True)
    knn_mean = d.mean(axis=1)
    med = np.median(knn_mean)
    mad = np.median(np.abs(knn_mean - med)) + 1e-12
    z = (knn_mean - med) / (1.4826 * mad)
    return z


def write_dashboard(out_dir: str, title: str, items: list[tuple[str, str]]) -> None:
    lis = "\n".join([f'  <li><a href="{fn}">{label}</a></li>' for label, fn in items])
    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; }}
h1 {{ font-size: 22px; }}
ul {{ line-height: 1.8; }}
.small {{ color: #555; font-size: 12px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="small">Pack fast. Pas de sky map sans ra/dec.</p>
<ul>
{lis}
</ul>
</body>
</html>
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["galaxy_candidates", "vari_summary"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-regions", type=int, default=0, help="Process only the first N input chunks (0 = all)")
    ap.add_argument("--max-rows", type=int, default=0, help="Limit rows per chunk (0 = all)")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if args.max_regions and args.max_regions > 0:
        paths = paths[: args.max_regions]
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    frames = []
    for path in paths:
        m = re.search(r"_(\d{6}-\d{6})", os.path.basename(path))
        rid = m.group(1) if m else os.path.basename(path).replace(".csv.gz", "")
        df = read_ecsv_gz(path)
        if args.max_rows and args.max_rows > 0:
            df = df.head(args.max_rows)
        df["region_id"] = rid
        df["__source_file"] = os.path.basename(path)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)

    os.makedirs(args.out, exist_ok=True)
    feature_cols = infer_feature_columns(all_df, "source_id")
    X_p, emb2 = fit_transform_features(all_df, feature_cols, pca_dim=10)
    z = knn_score(X_p, k=15)
    score = rank01(z)

    scored = all_df.copy()
    scored["score_z"] = z
    scored["score"] = score
    scored["emb_x"] = emb2[:, 0]
    scored["emb_y"] = emb2[:, 1]
    scored.to_csv(os.path.join(args.out, f"{args.kind}_scored_all.csv.gz"), index=False, compression="gzip")

    for rid, sub in scored.groupby("region_id", sort=True):
        reg_dir = os.path.join(args.out, f"region_{rid}")
        os.makedirs(reg_dir, exist_ok=True)

        sub.to_csv(os.path.join(reg_dir, "scored.csv.gz"), index=False, compression="gzip")

        plt.figure(figsize=(8, 6))
        plt.scatter(sub["emb_x"], sub["emb_y"], s=6, c=sub["score"])
        plt.title(f"{args.kind} embedding (PCA2) region {rid}")
        plt.xlabel("emb_x")
        plt.ylabel("emb_y")
        plt.tight_layout()
        plt.savefig(os.path.join(reg_dir, "01_embedding_pca.png"), dpi=160)
        plt.close()

        top = sub.nlargest(min(20, len(sub)), "score")[["source_id", "score"]]
        with open(os.path.join(reg_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "kind": args.kind,
                    "region_id": rid,
                    "n_rows": int(len(sub)),
                    "feature_cols": feature_cols,
                    "top_anomalies": top.to_dict(orient="records"),
                },
                f,
                indent=2,
            )

        write_dashboard(
            reg_dir,
            f"{args.kind} region {rid}",
            [
                ("Scored table (csv.gz)", "scored.csv.gz"),
                ("Embedding PCA (PNG)", "01_embedding_pca.png"),
                ("Summary (JSON)", "summary.json"),
            ],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
