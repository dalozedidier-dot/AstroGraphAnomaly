Notebooks
=========

Objectif
--------
Avoir 2 ou 3 notebooks Colab qui restent alignés avec le code et les workflows CI.

Notebooks recommandés
--------------------

1) `notebooks/colab_workflow.ipynb`
- Run offline CSV de test (smoke-friendly).
- Extras optionnels (désactivés par défaut) : A→H, graph explorer, region pack fast.

2) `notebooks/colab_A_to_H_suite.ipynb`
- Démo complète de la galerie A→H (+ HR/CMD).
- Plotly offline : les HTML embarquent plotly.js (pas de CDN).

3) `notebooks/colab_region_pack_fast.ipynb`
- Nouveau : runner “zones” tabulaires (GalaxyCandidates / VariSummary).
- Produit des dashboards HTML par région + embeddings PCA2.

Pourquoi certains notebooks ne sont plus “premiers”
---------------------------------------------------
Le repo contient aussi des notebooks historiques (showcase, plots_gallery, etc.). Ils restent utiles
mais peuvent dériver plus vite du pipeline principal. Les trois notebooks ci-dessus sont ceux à maintenir.

Points d’attention
-----------------

1) Dépendances “viz” (Plotly, PyVis, GIF, UMAP)
- La galerie A→H dépend de `requirements_viz.txt`.
- Si tu vois dans les HTML : « Plotly not installed » ou « PyVis not installed », c’est que la cellule `pip install -r requirements_viz.txt` n’a pas été exécutée (ou a échoué).

2) Ouverture des HTML dans Colab
- Les fichiers `06_explorer_dashboard.html`, `02_celestial_sphere_3d.html`, `03_network_explorer.html` utilisent des chemins relatifs (PNG, GIF, plotly.js, etc.).
- Dans Colab, l’affichage direct peut casser les liens (sandbox). Recommandation : zipper `results/<run>/viz_a_to_h/` puis télécharger et ouvrir localement.
- Alternative : lancer un mini serveur HTTP local (`python -m http.server`) et ouvrir le lien dans un nouvel onglet.

3) HR/CMD outliers
- Le plot HR/CMD requiert au minimum : `phot_g_mean_mag`, `parallax`, et une couleur `bp_rp`.
- Si `bp_rp` est absent mais que `phot_bp_mean_mag` et `phot_rp_mean_mag` existent, le script peut calculer `bp_rp`.
- Si aucune couleur n’est dispo, le module affiche un placeholder (ce qui est normal sur certains CSV “samples”).
