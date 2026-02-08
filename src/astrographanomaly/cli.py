import argparse
from .pipeline import run_pipeline

def build_parser():
    p = argparse.ArgumentParser(prog="astrographanomaly", description="AstroGraphAnomaly workflow (Gaia/CSV → graph → anomalies → LIME → exports).")
    sp = p.add_subparsers(dest="mode", required=True)

    # CSV
    pcsv = sp.add_parser("csv", help="Run pipeline from an input CSV (offline friendly)")
    pcsv.add_argument("--in-csv", required=True)
    pcsv.add_argument("--out", required=True)

    # Gaia
    pgaia = sp.add_parser("gaia", help="Run pipeline by querying Gaia DR3 (requires network)")
    pgaia.add_argument("--ra", type=float, required=True)
    pgaia.add_argument("--dec", type=float, required=True)
    pgaia.add_argument("--radius-deg", type=float, default=0.5)
    pgaia.add_argument("--limit", type=int, default=2000)
    pgaia.add_argument("--out", required=True)

    # Hubble (optional; may require additional configuration)
    phub = sp.add_parser("hubble", help="Run pipeline from a Hubble/HST-like CSV (or optional MAST query if implemented)")
    phub.add_argument("--in-csv", required=True)
    phub.add_argument("--out", required=True)

    for pp in (pcsv, pgaia, phub):
        pp.add_argument("--engine", default="isolation_forest", choices=["isolation_forest","lof","ocsvm","robust_zscore","pineforest"])
        pp.add_argument("--threshold-strategy", default="contamination", choices=["contamination","percentile","top_k","score"])
        pp.add_argument("--contamination", type=float, default=0.05)
        pp.add_argument("--percentile", type=float, default=95.0)
        pp.add_argument("--top-k", type=int, default=50)
        pp.add_argument("--score-threshold", type=float, default=1.0)

        pp.add_argument("--knn-k", type=int, default=8)
        pp.add_argument("--features-mode", default="extended", choices=["basic","extended"])

        pp.add_argument("--plots", action="store_true")
        pp.add_argument("--explain-top", type=int, default=0)
        pp.add_argument("--lime-num-features", type=int, default=8)

        pp.add_argument("--seed", type=int, default=42)

    return p

def main(argv=None):
    args = build_parser().parse_args(argv)

    common = dict(
        mode=args.mode,
        out_dir=args.out,
        engine=args.engine,
        threshold_strategy=args.threshold_strategy,
        contamination=args.contamination,
        percentile=args.percentile,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        knn_k=args.knn_k,
        features_mode=args.features_mode,
        explain_top=args.explain_top,
        lime_num_features=args.lime_num_features,
        plots=args.plots,
        seed=args.seed,
    )

    if args.mode == "gaia":
        run_pipeline(ra=args.ra, dec=args.dec, radius_deg=args.radius_deg, limit=args.limit, **common)
    else:
        run_pipeline(in_csv=args.in_csv, **common)
