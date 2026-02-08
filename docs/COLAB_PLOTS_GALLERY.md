# Colab: afficher et exporter les plots

Ce patch ajoute un notebook Colab minimal et un script utilitaire pour:

1. Afficher tous les PNG produits par un run (plots/ ou viz_a_to_h/).
2. Exporter un petit set d'images "headline" dans `screenshots/` pour le README.
3. Générer `screenshots/index.html` pour parcourir la galerie localement.

## Notebook

Fichier: `notebooks/colab_plots_gallery.ipynb`

Il permet:
- de monter Google Drive
- de pointer vers un dossier (Drive ou repo)
- d'afficher tous les PNG trouvés (y compris sous-dossiers)
- de zipper un dossier et le télécharger

## Script: collecter des screenshots README

Après un run (ex: `results/run_csv`), exécute:

```bash
python tools/collect_readme_screenshots.py --run-dir results/run_csv --out screenshots
```

Pour lister ce qui est disponible:

```bash
python tools/collect_readme_screenshots.py --run-dir results/run_csv --list
```

La sélection par défaut copie, si présents:
- graph_communities_anomalies.png
- ra_dec_score.png
- pca_2d.png
- top_anomalies_scores.png
- 01_hidden_constellations_sky.png
- 09_umap_cosmic_cloud.png
