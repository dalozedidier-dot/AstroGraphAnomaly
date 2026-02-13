#!/usr/bin/env python3
"""
Fetch Gaia DR3 gaia_source columns for a list of source_id using the Gaia Archive TAP service.

Usage:
  python tools/fetch_gaia_source_tap.py \
    --ids-csv ids.csv \
    --out gaia_source_enrich.csv.gz \
    --cols "source_id,ra,dec,parallax,pmra,pmdec,phot_g_mean_mag,bp_rp,ruwe" \
    --chunk-size 2000

Endpoint:
  https://gea.esac.esa.int/tap-server/tap/sync
"""
from __future__ import annotations

import argparse
import time
from typing import Iterable

import pandas as pd
import requests

TAP_SYNC = "https://gea.esac.esa.int/tap-server/tap/sync"


def chunks(lst: list[int], n: int) -> Iterable[list[int]]:
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def tap_query(adql: str) -> pd.DataFrame:
    r = requests.post(
        TAP_SYNC,
        data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": adql},
        timeout=300,
    )
    r.raise_for_status()
    from io import StringIO

    return pd.read_csv(StringIO(r.text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", required=True)
    ap.add_argument("--chunk-size", type=int, default=2000)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()

    ids_df = pd.read_csv(args.ids_csv)
    if "source_id" not in ids_df.columns:
        raise SystemExit("ids-csv must contain a source_id column")
    ids = ids_df["source_id"].dropna().astype("int64").tolist()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    if "source_id" not in cols:
        cols = ["source_id"] + cols

    out_frames = []
    for i, chunk in enumerate(chunks(ids, args.chunk_size), start=1):
        in_list = ",".join(str(x) for x in chunk)
        adql = f"SELECT {', '.join(cols)} FROM gaiadr3.gaia_source WHERE source_id IN ({in_list})"
        df = tap_query(adql)
        out_frames.append(df)
        print(f"chunk {i}: rows={len(df)}")
        if args.sleep > 0:
            time.sleep(args.sleep)

    out = pd.concat(out_frames, ignore_index=True).drop_duplicates(subset=["source_id"])
    out.to_csv(args.out, index=False, compression="gzip")
    print(f"wrote: {args.out} rows={len(out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
