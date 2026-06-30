# GitHub Pages relié aux workflows

La page publique n’est pas une vitrine statique séparée du projet.
Elle est générée par le workflow :

```text
.github/workflows/pages.yml
```

Le workflow fait quatre choses :

```text
GitHub Actions
→ run_workflow.py sur data/sample_gaia_like.csv
→ résultats dans results/<run_name>
→ tools/plotly_3d_report.py + tools/graph_viz.py
→ tools/build_pages_site.py
→ déploiement GitHub Pages
```

## Ce que la page affiche

La page publiée montre :

- le dernier run produit par le workflow Pages ;
- le moteur utilisé : `ensemble`, `isolation_forest`, `lof`, `ocsvm`, `robust_zscore` ou `pineforest` ;
- la stratégie de seuil ;
- les compteurs du run : lignes, anomalies, nœuds, arêtes ;
- les artefacts réellement générés : `report.html`, `summary.json`, `manifest.json`, `top_anomalies.csv`, `scored.csv`, `graph_full.graphml`, `graph_topk.graphml`, plots ;
- les vues 3D interactives générées par le workflow : `viz_plotly_3d/*` et `viz_graph_force/*` ;
- l’inventaire des workflows présents dans `.github/workflows`.

## Vues 3D interactives

Le workflow Pages génère aussi des fichiers HTML interactifs ouvrables directement depuis GitHub Pages :

```text
results/<run_name>/viz_plotly_3d/index.html
results/<run_name>/viz_plotly_3d/01_star_cloud_xyz.html
results/<run_name>/viz_plotly_3d/02_celestial_sphere.html
results/<run_name>/viz_plotly_3d/03_graph_topk_3d.html
results/<run_name>/viz_graph_force/plotly_topk_dim3.html
```

Ces fichiers sont produits après le pipeline par :

```bash
python tools/plotly_3d_report.py --run-dir results/<run_name>
python tools/graph_viz.py --run-dir results/<run_name> --backend plotly --dim 3 --graph topk --max-nodes 400
```

Les HTML Plotly embarquent désormais leur JavaScript directement dans le fichier. C’est plus lourd qu’un lien CDN, mais plus fiable sur GitHub Pages, dans les artefacts GitHub Actions et sur les réseaux qui bloquent `cdn.plot.ly`.

## Réglage GitHub

Dans GitHub :

```text
Settings → Pages → Source → GitHub Actions
```

Ensuite, lancer :

```text
Actions → deploy_pages_from_workflow_outputs → Run workflow
```

ou pousser une modification sur `main`.

## Pourquoi cette version est différente d’une simple page `models.html`

Une page `models.html` écrite à la main peut rapidement devenir décorative et déconnectée du dépôt.
Ici, la page est reconstruite depuis :

```text
run_workflow.py
.github/workflows/*.yml
results/<run_name>/*
```

Donc elle suit le fonctionnement réel du projet : workflow, pipeline, modèles, seuils et artefacts.

## Commande équivalente locale

```bash
python run_workflow.py \
  --mode csv \
  --in-csv data/sample_gaia_like.csv \
  --out results/pages_demo \
  --engine ensemble \
  --threshold-strategy top_k \
  --top-k 20 \
  --knn-k 8 \
  --features-mode extended \
  --plots \
  --explain-top 5

python tools/plotly_3d_report.py --run-dir results/pages_demo
python tools/graph_viz.py --run-dir results/pages_demo --backend plotly --dim 3 --graph topk --max-nodes 400

python tools/build_pages_site.py \
  --run-dir results/pages_demo \
  --run-name pages_demo \
  --out _site
```

Ouvrir ensuite :

```text
_site/index.html
```
