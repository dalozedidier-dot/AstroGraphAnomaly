# Colab Test maximal 2026

Ouvre `notebooks/colab_test_maximal_2026.ipynb` dans Google Colab.

Ce notebook:
- charge raw.csv, scored.csv, top_anomalies.csv, et optionnellement graph_full.graphml
- cross-match SIMBAD sur top anomalies (10 arcsec)
- genere des figures Plotly interactives + export PNG HD via kaleido
- embeddings UMAP et t-SNE
- optionnel: smoke-test torch-geometric sur graph_full si installe

Sorties:
- outputs/ : HTML interactifs + CSV crossmatch
- screenshots/ : PNG HD correspondants
- outputs_all.zip : bundle complet a telecharger
