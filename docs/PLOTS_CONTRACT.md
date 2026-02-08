PLOTS_CONTRACT — Suite complète (curated)
========================================

Contrat (8 PNG)
---------------
- score_hist.png
- ra_dec_score.png
- top_anomalies_scores.png
- mean_features_anom_vs_normal.png
- pca_2d.png
- mag_vs_distance.png
- graph_communities_anomalies.png
- cmd_bp_rp_vs_g.png

Post-run
--------
python tools/full_plots_suite.py --scored results/<run>/scored.csv --graph results/<run>/graph_full.graphml --out results/<run>/plots --top-k 30 --write-enriched

Wrapper (1 commande)
-------------------
python tools/run_full.py csv --in-csv data/sample_gaia_like.csv --out results/run_full --top-k 30 --knn-k 8 --engine isolation_forest
