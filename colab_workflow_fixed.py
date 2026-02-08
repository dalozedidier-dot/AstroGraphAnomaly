"""Colab/CLI helper script (workflow-only) for AstroGraphAnomaly.

This script is intended for Colab or any environment where you cloned the repo.
It detects `workflow.py` or `run_workflow.py` and runs an offline CSV smoke test.

Repo: https://github.com/dalozedidier-dot/AstroGraphAnomaly.git
"""

from __future__ import annotations
import sys, subprocess
from pathlib import Path

def detect_entrypoint() -> str:
    if Path("workflow.py").exists():
        return "workflow.py"
    if Path("run_workflow.py").exists():
        return "run_workflow.py"
    raise FileNotFoundError("No entrypoint found: workflow.py or run_workflow.py")

def run_offline_csv(out_dir="results/colab_csv", top_k=20, explain_top=5, plots=True) -> None:
    ep = detect_entrypoint()
    if ep == "workflow.py":
        cmd = [sys.executable, ep, "csv",
               "--in-csv", "data/sample_gaia_like.csv",
               "--out", out_dir,
               "--top-k", str(top_k),
               "--explain-top", str(explain_top)]
        if plots:
            cmd.append("--plots")
    else:
        cmd = [sys.executable, ep,
               "--mode", "csv",
               "--in-csv", "data/sample_gaia_like.csv",
               "--out", out_dir,
               "--top-k", str(top_k),
               "--explain-top", str(explain_top)]
        if plots:
            cmd.append("--plots")

    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    run_offline_csv()
