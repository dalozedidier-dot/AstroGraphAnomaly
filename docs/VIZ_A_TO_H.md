A→H Visualization Suite (Top-of-top)
===================================

Objectif
--------
Générer une galerie "artistique + scientifique" à partir des artefacts d'un run AstroGraphAnomaly.

Cette version est robuste en CI: une viz qui échoue ne casse pas la suite complète. Elle supporte aussi
un encodage couleur multi-contrainte basé sur des colonnes `phi_*` (ou `score_*`) si tu produis un score composite
ou des contributions par engine.

Entrées attendues
-----------------
- `scored.csv` (obligatoire)
- `graph_full.graphml` ou `graph_topk.graphml` (recommandé pour C)
- `explanations.jsonl` (optionnel, pour D; sinon fallback z-scores robustes)
- Colonnes utiles dans `scored.csv` (si disponibles): `ra`, `dec`, `pmra`, `pmdec`, `phot_g_mean_mag`, `bp_rp`, etc.

Sorties (dans `<run>/viz_a_to_h/`)
---------------------------------
A) `01_hidden_constellations_sky.png`
B) `02_celestial_sphere_3d.html` (Plotly interactif)
C) `03_network_explorer.html` (PyVis interactif)
D) `04_explainability_heatmap.png` + `05_feature_interaction_heatmap.png`
E) `06_explorer_dashboard.html` (index)
F) `07_proper_motion_trails.gif`
G) `08_feature_biocubes.html` (Plotly 3D)
H) `09_umap_cosmic_cloud.png` + `10_umap_cosmic_cloud.html`
I) `11_hr_cmd_outliers.png` + `12_hr_cmd_outliers.html`

Installation
------------
Base:
- `pip install -r requirements.txt`

Viz suite (recommandé):
- `pip install -r requirements_viz.txt`

Génération
----------
Depuis la racine du repo:

```bash
python tools/viz_a_to_h_suite.py   --run-dir results/<run>   --scored results/<run>/scored.csv   --graph  results/<run>/graph_full.graphml   --explain results/<run>/explanations.jsonl
```

Multi-couleurs par incohérences (phi_*)
---------------------------------------
Pré-requis: des colonnes de type `phi_<engine>` dans `scored.csv` (valeurs 0..1 recommandées).

Exemple "dominant": la couleur reflète la contrainte dominante (catégories), l'intensité reflète l'amplitude.

```bash
python tools/viz_a_to_h_suite.py   --run-dir results/<run>   --scored results/<run>/scored.csv   --graph  results/<run>/graph_full.graphml   --color-mode dominant_phi   --phi-prefix phi_   --phi-weights "graph=2.0,lof=1.0,ocsvm=1.0"
```

Exemple "RGB": mélange de 3 contraintes dans les canaux RGB (plus "blend").

```bash
python tools/viz_a_to_h_suite.py   --run-dir results/<run>   --scored results/<run>/scored.csv   --graph  results/<run>/graph_full.graphml   --color-mode rgb_phi   --phi-prefix phi_   --rgb-phis "graph,lof,ocsvm"
```

Notes
-----
- Les HTML interactifs (Plotly/PyVis) doivent être ouverts localement dans un navigateur (ou servis via un serveur local). GitHub ne rend pas le JS.
- Si `ra/dec` manquent, la sky map (A) et la sphère (B) basculent sur un placeholder informatif.
