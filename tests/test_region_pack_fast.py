from __future__ import annotations

from pathlib import Path
import subprocess


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def test_region_pack_fast_on_zone_fixtures(tmp_path: Path) -> None:
    raw = Path("data/region_pack/raw")
    assert raw.exists()

    gc = raw / "GalaxyCandidates_014046-015369.csv.gz"
    vs = raw / "VariSummary_012598-014045.csv.gz"
    assert gc.exists()
    assert vs.exists()

    out_gc = tmp_path / "gc"
    out_vs = tmp_path / "vs"

    _run(
        [
            "python",
            "tools/region_pack_fast.py",
            "--kind",
            "galaxy_candidates",
            "--inputs",
            str(gc),
            "--out",
            str(out_gc),
            "--engine",
            "robust_zscore",
            "--top-k",
            "25",
            "--max-rows",
            "1500",
        ]
    )
    _run(
        [
            "python",
            "tools/region_pack_fast.py",
            "--kind",
            "vari_summary",
            "--inputs",
            str(vs),
            "--out",
            str(out_vs),
            "--engine",
            "robust_zscore",
            "--top-k",
            "25",
            "--max-rows",
            "1500",
        ]
    )

    assert (out_gc / "galaxy_candidates_scored_all.csv.gz").exists()
    assert (out_vs / "vari_summary_scored_all.csv.gz").exists()

    # one per-region output for each
    assert list(out_gc.glob("region_*/scored.csv.gz"))
    assert list(out_vs.glob("region_*/scored.csv.gz"))
