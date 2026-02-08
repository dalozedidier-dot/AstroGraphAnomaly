#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Max runner (workflow-first)

But:
- Lancer le workflow avec la couverture maximale *possible* (engines × threshold strategies)
- En restant portable: détecte automatiquement les flags supportés via --help
- Génère un CSV offline enrichi (bp_rp) pour activer le CMD même sans Gaia
- Option Gaia: RUN_GAIA=1 (réseau/quota)
- Post-analyse: métriques graphe avancées + plots diagnostics
- Validation: contrat d'artefacts par run

Usage:
  python tools/run_max_all.py
  RUN_GAIA=1 python tools/run_max_all.py

Env vars:
  AGA_TOPK=50
  AGA_EXPLAIN_TOP=15
  AGA_KNNK=10
  RUN_GAIA=1
  AGA_RA=266.4051
  AGA_DEC=-28.936175
  AGA_RADIUS=0.3
  AGA_LIMIT=1200
"""

from __future__ import annotations

import os
import re
import sys
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Caps:
    engine: bool
    threshold_strategy: bool
    features_mode: bool
    explain_top: bool
    knn_k: bool
    plots: bool
    plots_level: bool


def detect_entrypoint() -> str:
    if Path("workflow.py").exists():
        return "workflow.py"
    if Path("run_workflow.py").exists():
        return "run_workflow.py"
    raise FileNotFoundError("Aucun entrypoint: workflow.py ou run_workflow.py")


def get_help(ep: str) -> str:
    try:
        return subprocess.check_output([sys.executable, ep, "--help"], stderr=subprocess.STDOUT, text=True)
    except Exception:
        return ""


def detect_caps(help_txt: str) -> Caps:
    def has(flag: str) -> bool:
        return flag in help_txt

    return Caps(
        engine=has("--engine"),
        threshold_strategy=has("--threshold-strategy"),
        features_mode=has("--features-mode"),
        explain_top=has("--explain-top"),
        knn_k=has("--knn-k"),
        plots=has("--plots"),
        plots_level=has("--plots-level"),
    )


def ensure_bp_rp_csv(base_csv: Path, out_csv: Path) -> None:
    """Offline CSV enriched with bp_rp to activate CMD plot."""
    df = pd.read_csv(base_csv)
    if "bp_rp" not in df.columns:
        rng = np.random.default_rng(42)
        df["bp_rp"] = np.clip(rng.normal(1.5, 0.6, len(df)), -0.5, 4.5)
    out_csv.write_text(df.to_csv(index=False), encoding="utf-8")


def build_cmd(
    ep: str,
    mode: str,
    out_dir: Path,
    caps: Caps,
    in_csv: Optional[Path] = None,
    ra: Optional[float] = None,
    dec: Optional[float] = None,
    radius_deg: float = 0.3,
    limit: int = 1200,
    engine: str = "isolation_forest",
    threshold_strategy: str = "top_k",
    top_k: int = 50,
    explain_top: int = 15,
    knn_k: int = 10,
    features_mode: str = "extended",
    plots: bool = True,
    plots_level: str = "basic",
) -> List[str]:

    # workflow.py (subcommands) vs run_workflow.py (--mode)
    if ep == "workflow.py":
        cmd = [sys.executable, ep, mode]
        if mode == "csv":
            if in_csv is None:
                raise ValueError("in_csv requis pour mode csv")
            cmd += ["--in-csv", str(in_csv)]
        elif mode == "gaia":
            if ra is None or dec is None:
                raise ValueError("ra/dec requis pour mode gaia")
            cmd += ["--ra", str(ra), "--dec", str(dec), "--radius-deg", str(radius_deg), "--limit", str(limit)]
        cmd += ["--out", str(out_dir)]
    else:
        cmd = [sys.executable, ep, "--mode", mode]
        if mode == "csv":
            if in_csv is None:
                raise ValueError("in_csv requis pour mode csv")
            cmd += ["--in-csv", str(in_csv)]
        elif mode == "gaia":
            if ra is None or dec is None:
                raise ValueError("ra/dec requis pour mode gaia")
            cmd += ["--ra", str(ra), "--dec", str(dec), "--radius-deg", str(radius_deg), "--limit", str(limit)]
        cmd += ["--out", str(out_dir)]

    # Optional flags guarded by caps
    if caps.engine:
        cmd += ["--engine", engine]
    if caps.threshold_strategy:
        cmd += ["--threshold-strategy", threshold_strategy]

    cmd += ["--top-k", str(int(top_k))]

    if caps.explain_top:
        cmd += ["--explain-top", str(int(explain_top))]
    if caps.knn_k:
        cmd += ["--knn-k", str(int(knn_k))]
    if caps.features_mode:
        cmd += ["--features-mode", features_mode]

    if plots and caps.plots:
        cmd += ["--plots"]
    if caps.plots_level:
        cmd += ["--plots-level", plots_level]

    return cmd


def run(cmd: List[str]) -> int:
    print("RUN:", " ".join(cmd))
    try:
        subprocess.check_call(cmd)
        return 0
    except subprocess.CalledProcessError as e:
        return int(e.returncode)


def main() -> None:
    ep = detect_entrypoint()
    help_txt = get_help(ep)
    caps = detect_caps(help_txt)

    print("Entrypoint:", ep)
    print("Caps:", caps)

    data_dir = Path("data")
    base_csv = data_dir / "sample_gaia_like.csv"
    if not base_csv.exists():
        raise FileNotFoundError(f"Missing offline dataset: {base_csv}")

    csv_bp = data_dir / "sample_gaia_like_with_bp_rp.csv"
    if not csv_bp.exists():
        ensure_bp_rp_csv(base_csv, csv_bp)

    top_k = int(os.environ.get("AGA_TOPK", "50"))
    explain_top = int(os.environ.get("AGA_EXPLAIN_TOP", "15"))
    knn_k = int(os.environ.get("AGA_KNNK", "10"))

    engines = ["isolation_forest", "lof", "ocsvm", "robust_zscore"]
    thresholds = ["top_k", "percentile", "contamination"]

    if caps.engine:
        m = re.search(r"--engine\s+\{([^}]+)\}", help_txt)
        if m:
            engines = [x.strip() for x in m.group(1).split(",") if x.strip()]

    if caps.threshold_strategy:
        m = re.search(r"--threshold-strategy\s+\{([^}]+)\}", help_txt)
        if m:
            thresholds = [x.strip() for x in m.group(1).split(",") if x.strip()]

    out_root = Path("results/max_runs")
    out_root.mkdir(parents=True, exist_ok=True)

    runs = []

    # Offline sweep (CSV)
    for eng in (engines if caps.engine else ["isolation_forest"]):
        for thr in (thresholds if caps.threshold_strategy else ["top_k"]):
            out_dir = out_root / f"csv__{eng}__{thr}"
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_cmd(
                ep=ep,
                mode="csv",
                out_dir=out_dir,
                caps=caps,
                in_csv=csv_bp,
                engine=eng,
                threshold_strategy=thr,
                top_k=top_k,
                explain_top=explain_top,
                knn_k=knn_k,
                features_mode="extended",
                plots=True,
                plots_level="basic",
            )

            rc = run(cmd)
            runs.append({"mode": "csv", "engine": eng, "threshold": thr, "out_dir": str(out_dir), "rc": rc, "cmd": cmd})

    # Gaia optional
    if os.environ.get("RUN_GAIA", "0") == "1":
        out_dir = out_root / "gaia__isolation_forest__top_k"
        out_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_cmd(
            ep=ep,
            mode="gaia",
            out_dir=out_dir,
            caps=caps,
            ra=float(os.environ.get("AGA_RA", "266.4051")),
            dec=float(os.environ.get("AGA_DEC", "-28.936175")),
            radius_deg=float(os.environ.get("AGA_RADIUS", "0.3")),
            limit=int(os.environ.get("AGA_LIMIT", "1200")),
            engine="isolation_forest",
            threshold_strategy="top_k",
            top_k=top_k,
            explain_top=explain_top,
            knn_k=knn_k,
            features_mode="extended",
            plots=True,
            plots_level="basic",
        )
        rc = run(cmd)
        runs.append({"mode": "gaia", "engine": "isolation_forest", "threshold": "top_k", "out_dir": str(out_dir), "rc": rc, "cmd": cmd})

    # Post + validation
    from tools.post_analysis import post_analyze
    from tools.validate_outputs import validate_out_dir

    checks = []
    for r in runs:
        out_dir = Path(r["out_dir"])
        c = validate_out_dir(out_dir)
        checks.append({**r, **c})
        if c.get("ok"):
            post_analyze(out_dir)

    summary = {"entrypoint": ep, "caps": caps.__dict__, "runs": runs, "checks": checks}
    (out_root / "max_runs_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    df = pd.DataFrame(checks)
    cols = ["mode", "engine", "threshold", "rc", "ok", "n_plots", "has_cmd", "has_explanations", "has_prompts", "missing"]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
