"""Tests for the run report outputs (summary.json + self-contained report.html)."""

from __future__ import annotations

import json
from pathlib import Path

from astrographanomaly.pipeline import run_pipeline


def test_report_and_summary_generated(tmp_path: Path) -> None:
    out_dir = tmp_path / "run"
    res = run_pipeline(
        mode="csv",
        in_csv="data/sample_gaia_like.csv",
        out_dir=str(out_dir),
        engine="robust_zscore",
        threshold_strategy="top_k",
        top_k=20,
        knn_k=8,
        features_mode="extended",
        plots=True,
        seed=42,
    )

    # Artefacts referenced and present on disk.
    assert res["artefacts"]["summary"] == "summary.json"
    assert res["artefacts"]["report"] == "report.html"
    assert (out_dir / "summary.json").exists()
    assert (out_dir / "report.html").exists()

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["counts"]["n_rows"] == 1200
    assert summary["counts"]["n_anomalies"] == 20
    assert len(summary["top_anomalies"]) == 20
    assert {"min", "p50", "p99", "max"}.issubset(summary["score_stats"])

    # Report is self-contained: plots embedded as base64, no external asset refs.
    html = (out_dir / "report.html").read_text()
    assert "data:image/png;base64," in html
    assert "Top anomalies" in html


def test_report_without_plots_has_no_figures(tmp_path: Path) -> None:
    out_dir = tmp_path / "run_np"
    run_pipeline(
        mode="csv",
        in_csv="data/sample_gaia_like.csv",
        out_dir=str(out_dir),
        engine="robust_zscore",
        threshold_strategy="top_k",
        top_k=10,
        knn_k=8,
        features_mode="basic",
        plots=False,
        seed=42,
    )
    html = (out_dir / "report.html").read_text()
    # No plots dir => no embedded images, but the report still renders.
    assert "data:image/png;base64," not in html
    assert "Top anomalies" in html
