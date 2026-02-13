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
