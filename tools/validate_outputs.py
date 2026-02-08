#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AstroGraphAnomaly — Validation d'un out_dir

Contrôle:
- présence des artefacts principaux
- présence des plots
- présence des fichiers explainability (optionnels)
- invariants simples de colonnes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

EXPECTED = [
    "raw.csv",
    "scored.csv",
    "top_anomalies.csv",
    "graph_full.graphml",
    "graph_topk.graphml",
    "manifest.json",
]

def validate_out_dir(out_dir: str | Path) -> Dict[str, Any]:
    out_dir = Path(out_dir)

    missing = [f for f in EXPECTED if not (out_dir / f).exists()]
    ok = len(missing) == 0

    plots_dir = out_dir / "plots"
    plots = sorted([p.name for p in plots_dir.glob("*.png")]) if plots_dir.exists() else []
    has_cmd = (plots_dir / "cmd_bp_rp_vs_g.png").exists() if plots_dir.exists() else False

    # Explainability files (optional)
    has_explanations = (out_dir / "explanations.jsonl").exists()
    has_prompts = (out_dir / "llm_prompts.jsonl").exists()

    col_checks: Dict[str, Any] = {}
    scored_path = out_dir / "scored.csv"
    raw_path = out_dir / "raw.csv"

    if scored_path.exists():
        df = pd.read_csv(scored_path, nrows=5)
        col_checks["scored_has_source_id"] = "source_id" in df.columns
        col_checks["scored_has_anomaly_score"] = "anomaly_score" in df.columns
        col_checks["scored_has_anomaly_label"] = "anomaly_label" in df.columns

    if raw_path.exists():
        df = pd.read_csv(raw_path, nrows=5)
        col_checks["raw_has_ra"] = "ra" in df.columns
        col_checks["raw_has_dec"] = "dec" in df.columns

    return {
        "ok": ok,
        "missing": missing,
        "n_plots": int(len(plots)),
        "plots": plots,
        "has_cmd": bool(has_cmd),
        "has_explanations": bool(has_explanations),
        "has_prompts": bool(has_prompts),
        **col_checks,
    }
