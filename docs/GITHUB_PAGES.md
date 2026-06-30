# Publier la page web sur GitHub Pages

Le dépôt contient une page statique prête à publier :

- `index.html` : page d’accueil actuelle du toolkit.
- `models.html` : page dédiée aux modèles / engines AstroGraphAnomaly.
- `docs/MODELS.md` : version documentation Markdown.
- `.github/workflows/pages.yml` : workflow de déploiement automatique GitHub Pages.

## Méthode recommandée : GitHub Actions

1. Pousser les fichiers sur GitHub.
2. Aller dans **Settings → Pages**.
3. Dans **Build and deployment**, choisir **Source → GitHub Actions**.
4. Le workflow `deploy_pages` publiera automatiquement le site.

La page sera ensuite accessible via :

```text
https://<utilisateur>.github.io/<nom-du-repo>/models.html
```

## Quand le workflow se lance

Le workflow se déclenche automatiquement quand un de ces éléments change sur `main` :

- `index.html`
- `models.html`
- `docs/**`
- `README.md`
- `LICENSE`
- `.github/workflows/pages.yml`

Il peut aussi être lancé manuellement depuis **Actions → deploy_pages → Run workflow**.

## Ce que publie le workflow

Le workflow prépare un dossier `_site` contenant uniquement les fichiers utiles à la page publique :

```text
_site/
├── index.html
├── models.html
├── README.md
├── LICENSE
├── docs/
└── .nojekyll
```

Cela évite de publier inutilement tout le dépôt Python, les scripts, les notebooks ou les fichiers de test.

## Alternative simple

Il est aussi possible d’utiliser la méthode classique :

1. Aller dans **Settings → Pages**.
2. Choisir **Deploy from a branch**.
3. Choisir la branche `main` puis `/root`.

Mais pour ce dépôt, la méthode **GitHub Actions** est plus propre, car elle sépare la page publique du code scientifique.

## À vérifier

- Le fichier `index.html` est bien à la racine du dépôt.
- Le fichier `models.html` est bien à la racine du dépôt.
- Dans **Settings → Pages**, la source est bien réglée sur **GitHub Actions**.
- Le dépôt est public, ou GitHub Pages est autorisé pour le dépôt privé selon ton plan GitHub.
