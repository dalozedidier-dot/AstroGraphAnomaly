# Publier la page web sur GitHub Pages

Le dépôt contient maintenant une page statique prête à publier :

- `index.html` : page d’accueil actuelle du toolkit.
- `models.html` : page dédiée aux modèles / engines AstroGraphAnomaly.
- `docs/MODELS.md` : version documentation Markdown.

## Méthode simple

1. Pousser les fichiers sur GitHub.
2. Aller dans **Settings → Pages**.
3. Dans **Build and deployment**, choisir **Deploy from a branch**.
4. Choisir la branche `main` puis le dossier `/root`.
5. Enregistrer.

La page sera ensuite accessible via :

```text
https://<utilisateur>.github.io/<nom-du-repo>/models.html
```

## À vérifier

- Le fichier `index.html` est bien à la racine du dépôt.
- Le fichier `models.html` est bien à la racine du dépôt.
- Le dépôt est public, ou GitHub Pages est autorisé pour le dépôt privé selon ton plan GitHub.
