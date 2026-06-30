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
- l’inventaire des workflows présents dans `.github/workflows`.

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

python tools/build_pages_site.py \
  --run-dir results/pages_demo \
  --run-name pages_demo \
  --out _site
```

Ouvrir ensuite :

```text
_site/index.html
```
