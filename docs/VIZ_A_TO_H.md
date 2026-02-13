A→H Visualization Suite (Top-of-top)
===================================

Objectif
--------
Générer une galerie "artistique + scientifique" à partir des artefacts d'un run AstroGraphAnomaly,
en reproduisant la suite A→H définie dans la conversation.

Entrées attendues
-----------------
- scored.csv (obligatoire)
- graph_union.graphml ou graph_full.graphml (recommandé)
- explanations.jsonl (optionnel; sinon heatmap basée sur z-scores des features)

Sorties (dans <out>/viz_a_to_h/)
-------------------------------
A) 01_hidden_constellations_sky.png
B) 02_celestial_sphere_3d.html
C) 03_network_explorer.html
D) 04_explainability_heatmap.png + 05_feature_interaction_heatmap.png
E) 06_explorer_dashboard.html  (index qui relie tout)
F) 07_proper_motion_trails.gif
G) 08_feature_biocubes.html
H) 09_umap_cosmic_cloud.png + 10_umap_cosmic_cloud.html
I) 11_hr_cmd_outliers.png + 12_hr_cmd_outliers.html

Commande
--------
python tools/viz_a_to_h_suite.py \
  --run-dir results/<run> \
  --scored results/<run>/scored.csv \
  --graph  results/<run>/graph_full.graphml \
  --explain results/<run>/explanations.jsonl

Notes
-----
- Aucun packaging requis. Le script est autonome.
- Pour Colab: pip install -r requirements.txt puis pip install -r requirements_viz.txt

HTML Plotly offline
-------------------
Les fichiers Plotly (.html) embarquent plotly.js directement (pas de CDN). Tu peux les ouvrir localement (sans internet) depuis 06_explorer_dashboard.html.

Couleurs par incohérence (multi-colors)
---------------------------------------
Si ton scored.csv contient des colonnes phi_* (par ex. phi_graph, phi_lof, phi_ocsvm), tu peux colorer les points par incohérence dominante ou faire un mélange RGB.

Options CLI:
- --color-mode auto|score|dominant_phi|rgb_phi
- --phi-prefix (défaut: phi_)
- --phi-weights (ex: "graph=2.0,lof=1.0")
- --rgb-phis (ex: "graph,lof,ocsvm")

