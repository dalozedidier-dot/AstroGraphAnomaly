# GitHub Web + Colab (workflow)

## 1) Créer le dépôt (GitHub web)
- New repository (vide)
- Upload files → glisser/déposer le contenu de ce zip à la racine
- Commit

## 2) Lancer dans Colab
- Ouvrir `notebooks/colab_workflow.ipynb`
- Remplacer `<USER>/<REPO>` par votre repo
- Exécuter

## 3) Run offline (test)
- Utilise `data/sample_gaia_like.csv` (pas de réseau requis)

## 4) Run Gaia (réseau requis)
- `python -m astrographanomaly gaia ...`

## 5) Publier la page workflow-first

Pour publier une page web qui reflète vraiment les workflows du dépôt :

1. Aller dans `Settings → Pages`.
2. Choisir `Source → GitHub Actions`.
3. Aller dans `Actions`.
4. Lancer `deploy_pages_from_workflow_outputs`.

Ce workflow exécute le pipeline, génère `results/<run_name>`, construit `_site`, puis déploie GitHub Pages.
La page publiée ne dépend donc pas d’un `models.html` écrit à la main : elle est construite depuis les sorties réelles du workflow.


## 6) Vues 3D en ligne

Le workflow Pages ne se contente pas de copier `graph_full.graphml` ou `graph_topk.graphml`. Il transforme aussi les sorties du run en HTML interactif :

```text
runs/<run_name>/viz_plotly_3d/index.html
runs/<run_name>/viz_graph_force/plotly_topk_dim3.html
```

Ces pages sont liées depuis `index.html` et peuvent être ouvertes directement depuis GitHub Pages.
