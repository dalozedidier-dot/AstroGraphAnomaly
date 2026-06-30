# Modèles / moteurs AstroGraphAnomaly

AstroGraphAnomaly utilise le terme `engine` pour désigner les moteurs de scoring d’anomalies. Ces moteurs ne prouvent pas qu’un objet est astrophysiquement exceptionnel : ils classent des candidats qui méritent une inspection humaine.

## Engines disponibles

| Engine | Lecture principale | Points forts | Limites | Commande |
|---|---|---|---|---|
| `isolation_forest` | Rareté multivariée globale | Rapide, robuste, bon point de départ | Sensible au paramètre `contamination` | `--engine isolation_forest` |
| `lof` | Écart au voisinage local | Bon pour sous-populations et densités locales | Sensible au nombre de voisins | `--engine lof` |
| `ocsvm` | Frontière autour du comportement dominant | Non-linéaire, intéressant en comparaison | Coûteux sur gros volumes, sensible à l’échelle | `--engine ocsvm` |
| `robust_zscore` | Écarts robustes aux médianes | Simple, lisible, excellent garde-fou | Capte moins les anomalies relationnelles | `--engine robust_zscore` |
| `pineforest` | Forêt optionnelle via `coniferest` | Piste expérimentale/performance | Dépendance optionnelle | `--engine pineforest` |
| `ensemble` | Score composite multi-contraintes | Croise plusieurs moteurs + contrainte de graphe | Poids à documenter | `--engine ensemble` |

## Modes de features

- `basic` : degré, clustering, parallax, mouvements propres, magnitude, distance.
- `extended` : ajoute k-core, betweenness, communautés, points d’articulation, ponts.

## Stratégies de seuil

- `contamination` : fraction attendue d’anomalies.
- `percentile` : coupe au-dessus d’un percentile.
- `top_k` : garde les K meilleurs candidats.
- `score` : seuil numérique direct sur `anomaly_score`.

## Commande recommandée pour une démo

```bash
python run_workflow.py --mode csv \
  --in-csv data/sample_gaia_like.csv \
  --out results/demo_ensemble \
  --engine ensemble \
  --threshold-strategy top_k \
  --top-k 50 \
  --plots \
  --explain-top 10
```

## Message de prudence

Un score d’anomalie n’est pas une découverte astrophysique. C’est un signal de triage : il indique qu’une source mérite une inspection, une comparaison multi-modèles, puis une validation par données externes et expertise métier.
