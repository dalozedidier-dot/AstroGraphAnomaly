# Contributing

## Scope
Le projet est orienté **workflow-first** : exécution via scripts/CLI + notebooks Colab.
Pas d'obligation de packaging.

## PR checklist
- Le pipeline offline (CSV) passe en local/CI.
- Ajoute/maintient des artefacts reproductibles (`results/...`).
- Si tu ajoutes des plots, ils doivent être sauvegardés dans `results/<run>/plots/`.

## Style
- Code Python lisible, fonctions courtes.
- Pas de dépendances lourdes sans justification (CI/Colab).

## Issues
- Fournir la commande exacte, les 20 dernières lignes de log, et la config (`manifest.json` si disponible).
