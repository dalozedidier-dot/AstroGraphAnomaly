Pretty plots patch
=================

But:
- Garder un set de plots pertinent + lisible + esthétique, exporté systématiquement dans results/.../plots.

Fichier remplacé:
- src/astrographanomaly/reporting/plots.py

Nouveaux PNG exportés (si colonnes présentes):
- score_hist.png
- ra_dec_score.png
- mag_vs_distance.png
- cmd_bp_rp_vs_g.png (si bp_rp)
- mean_features_anom_vs_normal.png (si anomaly_label)
- top_anomalies_scores.png
- pca_2d.png
- graph_communities_anomalies.png (remplace/complète le graph plot précédent)

Intégration:
- aucune modification pipeline requise si le pipeline appelle déjà save_basic_plots() et save_graph_plot().
