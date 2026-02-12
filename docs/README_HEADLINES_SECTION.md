## Visual showcase

Les résultats incluent un dossier `viz_a_to_h/` avec une page d'entrée `06_explorer_dashboard.html`.
Cette page donne accès à la sphère céleste 3D, à l'explorateur réseau, aux biocubes de features, à l'UMAP interactive, aux heatmaps d'explicabilité, et au GIF des trajectoires de mouvement propre.

Aperçu (exemples d'images générées) :

![Hidden Constellations](screenshots/01_hidden_constellations_sky.png)
![UMAP](screenshots/09_umap_cosmic_cloud.png)
![Explainability](screenshots/04_explainability_heatmap.png)
![Feature interactions](screenshots/05_feature_interaction_heatmap.png)
![RA/Dec score](screenshots/ra_dec_score.png)
![Proper motion trails](screenshots/07_proper_motion_trails.gif)

Notes:
- Les HTML interactifs sont générés en mode 100% offline. Plotly.js est inliné dans chaque fichier HTML.
- Les captures ci-dessus peuvent être remplacées par celles d'un run plus récent en copiant depuis `results/<run>/viz_a_to_h/`.
