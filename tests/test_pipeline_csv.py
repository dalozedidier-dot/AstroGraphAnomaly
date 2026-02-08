import os
from pathlib import Path
import pandas as pd
from astrographanomaly.pipeline import run_pipeline

def test_pipeline_csv(tmp_path: Path):
    in_csv = Path("data/sample_gaia_like.csv")
    out_dir = tmp_path/"run"
    res = run_pipeline(
        mode="csv",
        in_csv=str(in_csv),
        out_dir=str(out_dir),
        engine="robust_zscore",
        threshold_strategy="top_k",
        top_k=20,
        knn_k=8,
        features_mode="extended",
        explain_top=0,
        plots=False,
        seed=42,
    )
    assert (out_dir/"raw.csv").exists()
    assert (out_dir/"scored.csv").exists()
    assert (out_dir/"top_anomalies.csv").exists()
    assert (out_dir/"manifest.json").exists()
    df = pd.read_csv(out_dir/"top_anomalies.csv")
    assert len(df) == 20
