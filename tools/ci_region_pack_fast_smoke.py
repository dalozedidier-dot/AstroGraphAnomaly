#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tools/ci_region_pack_fast_smoke.py

CI smoke test for the Region Pack fast pipeline, using the committed
data/region_pack/raw/*.csv.gz files.

This does NOT depend on remote Gaia services, so it remains stable in CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import subprocess


def _run(cmd: List[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=120)
    ap.add_argument("--engine", default="robust_zscore")
    ap.add_argument("--out-root", default="results/region_pack_fast_ci")
    args = ap.parse_args()

    raw = Path("data/region_pack/raw")
    if not raw.exists():
        raise SystemExit("Missing data/region_pack/raw (fixtures not present)")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    cat_maps = sorted(raw.glob("GalaxyCatalogueName_*.csv.gz"))

    # GalaxyCandidates
    gc_inputs = sorted(raw.glob("GalaxyCandidates_*.csv.gz"))
    if not gc_inputs:
        raise SystemExit("No GalaxyCandidates_*.csv.gz fixtures found")

    _run([
        "python",
        "tools/region_pack_fast.py",
        "--kind", "galaxy_candidates",
        "--inputs", *[str(p) for p in gc_inputs],
        "--out", str(out_root / "galaxy_candidates"),
        "--engine", str(args.engine),
        "--top-k", str(args.top_k),
        "--max-rows", str(args.max_rows),
        "--catalogue-maps", *[str(p) for p in cat_maps],
    ])

    # VariSummary
    vs_inputs = sorted(raw.glob("VariSummary_*.csv.gz"))
    if not vs_inputs:
        raise SystemExit("No VariSummary_*.csv.gz fixtures found")

    _run([
        "python",
        "tools/region_pack_fast.py",
        "--kind", "vari_summary",
        "--inputs", *[str(p) for p in vs_inputs],
        "--out", str(out_root / "vari_summary"),
        "--engine", str(args.engine),
        "--top-k", str(args.top_k),
        "--max-rows", str(args.max_rows),
    ])

    # Assertions
    agg1 = out_root / "galaxy_candidates" / "galaxy_candidates_scored_all.csv.gz"
    agg2 = out_root / "vari_summary" / "vari_summary_scored_all.csv.gz"
    if not agg1.exists() or not agg2.exists():
        raise SystemExit("Aggregate outputs missing")

    # Check one region output exists
    any_region = next((out_root / "galaxy_candidates").glob("region_*/scored.csv.gz"), None)
    if any_region is None:
        raise SystemExit("No per-region outputs found for galaxy_candidates")

    print("[ci_region_pack_fast_smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
