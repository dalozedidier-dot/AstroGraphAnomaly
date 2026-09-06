# GitHub Pages — tout voir sans télécharger

Site public (déjà déployé) :

**https://dalozedidier-dot.github.io/AstroGraphAnomaly/**

Ce n’est pas une copie du dépôt. Le workflow `deploy_pages_from_workflow_outputs` (`.github/workflows/pages.yml`) :

1. lance `run_workflow.py` sur `data/sample_gaia_like.csv`
2. génère les vues 3D (`tools/plotly_3d_report.py`, `tools/graph_viz.py`)
3. assemble un site statique avec `tools/build_pages_site.py`
4. déploie `_site/` sur GitHub Pages

## Ce qui s’ouvre dans le navigateur

| Page | URL |
| --- | --- |
| Tableau de bord | https://dalozedidier-dot.github.io/AstroGraphAnomaly/ |
| Rapport HTML du run | https://dalozedidier-dot.github.io/AstroGraphAnomaly/runs/pages_demo/report.html |
| Index des vues 3D | dans le dashboard, section *Vues 3D interactives* |

Plotly est **embarqué** dans les HTML : pas de CDN, pas de zip Actions à télécharger.

## Relancer une publication

Actions → **deploy_pages_from_workflow_outputs** → Run workflow.

Ou pousser sur `main` un changement dans `src/`, `tools/`, `run_workflow.py`, `data/sample_gaia_like.csv` ou les workflows.

Settings → Pages → Source = **GitHub Actions** (pas « Deploy from a branch »).
