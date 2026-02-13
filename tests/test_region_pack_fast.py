import os
import subprocess
import sys
from pathlib import Path


def test_region_pack_fast_smoke(tmp_path: Path) -> None:
    # Minimal smoke on one provided region chunk (no network).
    inp = Path("data/region_pack/raw/GalaxyCandidates_022411-022698.csv.gz")
    assert inp.exists(), f"missing test input: {inp}"

    out_dir = tmp_path / "region_fast"
    cmd = [
        sys.executable,
        "tools/run_regions_fast.py",
        "--kind",
        "galaxy_candidates",
        "--inputs",
        str(inp),
        "--out",
        str(out_dir),
        "--max-regions",
        "1",
        "--max-rows",
        "1000",
    ]
    subprocess.check_call(cmd)

    # Expected outputs
    assert (out_dir / "galaxy_candidates_scored_all.csv.gz").exists()
    reg_dir = out_dir / "region_022411-022698"
    assert (reg_dir / "scored.csv.gz").exists()
    assert (reg_dir / "01_embedding_pca.png").exists()
    assert (reg_dir / "summary.json").exists()
    assert (reg_dir / "index.html").exists()
