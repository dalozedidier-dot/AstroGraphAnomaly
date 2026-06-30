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

## 5) Publier la page des modèles

Le dépôt contient aussi une page statique :

- `index.html`
- `models.html`
- `docs/MODELS.md`
- `.github/workflows/pages.yml`

Pour la publier proprement :

1. Aller dans **Settings → Pages**.
2. Choisir **Source → GitHub Actions**.
3. Lancer ou attendre le workflow **deploy_pages**.

Page finale :

```text
https://<USER>.github.io/<REPO>/models.html
```

