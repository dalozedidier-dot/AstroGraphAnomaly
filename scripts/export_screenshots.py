#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copie un set de PNG depuis un run `results/<run>/plots/` vers `screenshots/`.

Usage:
  python scripts/export_screenshots.py --from results/run_csv/plots --to screenshots
"""

from __future__ import annotations
import argparse
from pathlib import Path
import shutil

DEFAULT = [
  "ra_dec_score.png",
  "score_hist.png",
  "top_anomalies_scores.png",
  "graph_communities_anomalies.png",
  "mag_vs_distance.png",
  "cmd_bp_rp_vs_g.png",
  "pca_2d.png",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", required=True)
    ap.add_argument("--to", dest="dst", default="screenshots")
    ap.add_argument("--max", type=int, default=6)
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    copied = 0
    for name in DEFAULT:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)
            copied += 1
            if copied >= args.max:
                break

    print(f"Copied {copied} files -> {dst}")

if __name__ == "__main__":
    main()
