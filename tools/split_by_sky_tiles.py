#!/usr/bin/env python3
"""
Split an enriched Gaia dataset into simple sky tiles (RA/Dec bins) to create new "regions".

This is a pragmatic alternative to HEALPix when you want zero extra dependencies.
If you later install healpy, you can swap to HEALPix tiling, but this works everywhere.

Input: a CSV (optionally .gz) that contains at least: source_id, ra, dec
Output: out/tiles/tile_raXXX_decYYY/scored.csv.gz (or any input table, not necessarily scored)

Usage:
  python tools/split_by_sky_tiles.py \
    --in results/galaxy_candidates_enriched_scored.csv.gz \
    --out results/galaxy_candidates_tiles \
    --ra-bins 24 \
    --dec-bins 12 \
    --min-rows 500

Notes:
- ra in degrees [0, 360), dec in degrees [-90, +90]
- Tiles are rectangular in (ra, dec). They are not equal-area, but very effective for diagnostics.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ra-bins", type=int, default=24)
    ap.add_argument("--dec-bins", type=int, default=12)
    ap.add_argument("--min-rows", type=int, default=200)
    args = ap.parse_args()

    df = pd.read_csv(args.inp, compression="infer")
    for col in ["source_id", "ra", "dec"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    ra = df["ra"].astype(float).to_numpy()
    dec = df["dec"].astype(float).to_numpy()

    ra_edges = np.linspace(0.0, 360.0, args.ra_bins + 1)
    dec_edges = np.linspace(-90.0, 90.0, args.dec_bins + 1)

    ra_idx = np.clip(np.digitize(ra, ra_edges) - 1, 0, args.ra_bins - 1)
    dec_idx = np.clip(np.digitize(dec, dec_edges) - 1, 0, args.dec_bins - 1)

    os.makedirs(args.out, exist_ok=True)

    df = df.copy()
    df["tile_ra"] = ra_idx
    df["tile_dec"] = dec_idx

    kept = 0
    for (i, j), sub in df.groupby(["tile_ra", "tile_dec"], sort=True):
        if len(sub) < args.min_rows:
            continue
        kept += 1
        tile_dir = os.path.join(args.out, f"tile_ra{i:02d}_dec{j:02d}")
        os.makedirs(tile_dir, exist_ok=True)
        sub.drop(columns=["tile_ra", "tile_dec"]).to_csv(
            os.path.join(tile_dir, "tile.csv.gz"),
            index=False,
            compression="gzip",
        )

    print(f"tiles kept: {kept}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
