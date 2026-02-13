"""Convenience runner (workflow-first, no editable install required).

Usage examples (from repo root):

  # Offline test run
  python run_workflow.py --mode csv --in-csv data/sample_gaia_like.csv --out results/run_csv --engine robust_zscore --threshold-strategy top_k --top-k 50 --plots --explain-top 10

  # Gaia run (network required)
  python run_workflow.py --mode gaia --ra 266.4051 --dec -28.936175 --radius-deg 0.5 --limit 2000 --out results/run_gaia --engine isolation_forest --plots --explain-top 10
"""

import argparse
import sys
from pathlib import Path

# Ensure `src/` is importable when running from repo root (GitHub web + Colab friendly)
repo_root = Path(__file__).resolve().parent
src_dir = repo_root / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from astrographanomaly.pipeline import run_pipeline  # noqa: E402

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["csv", "gaia", "hubble"])
    p.add_argument("--in-csv")
    p.add_argument("--ra", type=float)
    p.add_argument("--dec", type=float)
    p.add_argument("--radius-deg", type=float, default=0.5)
    p.add_argument("--limit", type=int, default=2000)
    p.add_argument("--out", required=True)

    p.add_argument(
        "--engine",
        default="isolation_forest",
        choices=["isolation_forest", "lof", "ocsvm", "robust_zscore", "pineforest", "ensemble"],
    )
    p.add_argument("--threshold-strategy", default="contamination", choices=["contamination","percentile","top_k","score"])
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--percentile", type=float, default=95.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--score-threshold", type=float, default=1.0)

    p.add_argument("--knn-k", type=int, default=8)
    p.add_argument("--features-mode", default="extended", choices=["basic","extended"])
    p.add_argument("--plots", action="store_true")
    p.add_argument("--explain-top", type=int, default=0)
    p.add_argument("--lime-num-features", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)

    # Ensemble options (used when --engine ensemble)
    p.add_argument(
        "--ensemble-engines",
        default="isolation_forest,lof,ocsvm",
        help="Comma-separated list of engines to combine (ensemble mode)",
    )
    p.add_argument(
        "--ensemble-weights",
        default="",
        help="Weights override, e.g. 'isolation_forest=1,lof=1,ocsvm=1,graph=1.5' (ensemble mode)",
    )
    p.add_argument(
        "--ensemble-no-graph-constraint",
        action="store_true",
        help="Disable the extra graph constraint (betweenness/articulation/bridge) in ensemble mode",
    )
    p.add_argument(
        "--ensemble-graph-weight",
        type=float,
        default=1.5,
        help="Weight for the graph constraint (ensemble mode)",
    )

    a = p.parse_args()

    run_pipeline(
        mode=a.mode,
        in_csv=a.in_csv,
        ra=a.ra, dec=a.dec, radius_deg=a.radius_deg, limit=a.limit,
        out_dir=a.out,
        engine=a.engine,
        threshold_strategy=a.threshold_strategy,
        contamination=a.contamination,
        percentile=a.percentile,
        top_k=a.top_k,
        score_threshold=a.score_threshold,
        knn_k=a.knn_k,
        features_mode=a.features_mode,
        explain_top=a.explain_top,
        lime_num_features=a.lime_num_features,
        plots=a.plots,
        seed=a.seed,
        ensemble_engines=a.ensemble_engines,
        ensemble_weights=a.ensemble_weights,
        ensemble_include_graph_constraint=(not bool(a.ensemble_no_graph_constraint)),
        ensemble_graph_weight=float(a.ensemble_graph_weight),
    )

if __name__ == "__main__":
    main()
