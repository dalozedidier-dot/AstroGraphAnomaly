#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run pipeline + regenerate full plots (one command).

Ne modifie pas le pipeline existant.
- exécute workflow.py (subcommands) si présent
- sinon exécute run_workflow.py (flags)

Puis appelle tools/full_plots_suite.py sur la sortie `scored.csv`.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    csv = sub.add_parser("csv")
    csv.add_argument("--in-csv", required=True)

    gaia = sub.add_parser("gaia")
    gaia.add_argument("--ra", type=float, required=True)
    gaia.add_argument("--dec", type=float, required=True)
    gaia.add_argument("--radius-deg", type=float, default=0.3)
    gaia.add_argument("--limit", type=int, default=1200)

    for p in (csv, gaia):
        p.add_argument("--out", required=True)
        p.add_argument("--top-k", type=int, default=30)
        p.add_argument("--knn-k", type=int, default=8)
        p.add_argument("--engine", default="isolation_forest")
        p.add_argument("--features-mode", default="extended")
        p.add_argument("--explain-top", type=int, default=5)
        p.add_argument("--write-enriched", action="store_true", default=True)

    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    has_workflow = Path("workflow.py").exists()
    has_run_workflow = Path("run_workflow.py").exists()
    if not has_workflow and not has_run_workflow:
        print("ERROR: workflow.py or run_workflow.py not found at repo root.")
        return 2

    if has_workflow:
        cmd = [sys.executable, "workflow.py", args.mode]
        if args.mode == "csv":
            cmd += ["--in-csv", args.in_csv]
        else:
            cmd += ["--ra", str(args.ra), "--dec", str(args.dec), "--radius-deg", str(args.radius_deg), "--limit", str(args.limit)]
        cmd += ["--out", str(out), "--top-k", str(args.top_k), "--knn-k", str(args.knn_k)]
        cmd += ["--engine", str(args.engine), "--features-mode", str(args.features_mode), "--explain-top", str(args.explain_top), "--plots"]
    else:
        cmd = [sys.executable, "run_workflow.py", "--mode", args.mode]
        if args.mode == "csv":
            cmd += ["--in-csv", args.in_csv]
        else:
            cmd += ["--ra", str(args.ra), "--dec", str(args.dec), "--radius-deg", str(args.radius_deg), "--limit", str(args.limit)]
        cmd += ["--out", str(out), "--top-k", str(args.top_k), "--knn-k", str(args.knn_k)]
        cmd += ["--engine", str(args.engine), "--features-mode", str(args.features_mode), "--explain-top", str(args.explain_top), "--plots"]

    print("PIPELINE:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print("Pipeline failed rc=", rc)
        return rc

    scored = out / "scored.csv"
    graph = out / "graph_full.graphml"
    plots = out / "plots"

    if not scored.exists():
        print("ERROR: scored.csv not found in", out)
        return 3

    cmd2 = [sys.executable, "tools/full_plots_suite.py", "--scored", str(scored), "--out", str(plots), "--top-k", str(args.top_k)]
    if graph.exists():
        cmd2 += ["--graph", str(graph)]
    if args.write_enriched:
        cmd2 += ["--write-enriched"]

    print("PLOTS:", " ".join(cmd2))
    return subprocess.call(cmd2)


if __name__ == "__main__":
    raise SystemExit(main())
