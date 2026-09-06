# AstroGraphAnomaly

[![ci](https://github.com/dalozedidier-dot/AstroGraphAnomaly/actions/workflows/ci.yml/badge.svg)](https://github.com/dalozedidier-dot/AstroGraphAnomaly/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Dépôt orienté **workflow** (GitHub web + Colab) :
- ingestion **Gaia** ou **CSV**
- construction de graphe **k-NN** sur la sphère céleste
- extraction de features (astro + métriques graphe avancées)
- détection d’anomalies (plusieurs engines + stratégies de seuil)
- explainability **LIME** + génération de **prompts LLM**
- exports : CSV / GraphML / PNG / manifest JSON / `report.html`

Package version: `0.2.0`.

This project is **astronomy** (Gaia catalogs + graphs). It is not ChaosTrace and not AstroOracle.

## Exécution recommandée

### Colab
Ouvrir `notebooks/colab_workflow.ipynb` (smoke + run offline).

Notebooks :
- `notebooks/colab_A_to_H_suite.ipynb` : galerie A→H (+ HR/CMD)
- `notebooks/colab_region_pack_fast.ipynb` : tests rapides GalaxyCandidates / VariSummary

### Local (repo root, sans install)
```bash
pip install -r requirements.txt
python run_workflow.py --mode csv --in-csv data/sample_gaia_like.csv --out results/run_csv --plots --explain-top 10
python run_workflow.py --mode gaia --ra 266.4051 --dec -28.936175 --radius-deg 0.5 --limit 2000 --out results/run_gaia --plots --explain-top 10
```

### Local (package)
```bash
pip install -e ".[dev]"
aga csv --in-csv data/sample_gaia_like.csv --out results/run_csv --plots --explain-top 10
```

## Résultats
Chaque run produit :
- `raw.csv`, `scored.csv`, `top_anomalies.csv`
- `graph_full.graphml`, `graph_topk.graphml`
- `summary.json`
- `report.html` (autonome, plots en base64)
- `plots/*.png` (si `--plots`)
- `explanations.jsonl` + `llm_prompts.jsonl` (si `--explain-top > 0`)
- `manifest.json`

## Engines
- `isolation_forest`
- `lof`
- `ocsvm`
- `robust_zscore`
- `pineforest` (extra `.[pineforest]`)
- `ensemble`

## Seuils
- `contamination`
- `percentile`
- `top_k`
- `score`

## Métriques graphe (mode `extended`)
- degree, clustering
- k-core
- betweenness (approx possible)
- communautés (Louvain si dispo, sinon greedy modularity)
- articulation points
- bridges (`incident_to_bridge`)

## Développement
```bash
pip install -e ".[dev]"
pytest -q
```
