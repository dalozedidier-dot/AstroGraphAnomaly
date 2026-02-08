# AstroGraphAnomaly (workflow-first)

Dépôt orienté **workflow** (GitHub web + Colab) :
- ingestion **Gaia** ou **CSV**
- construction de graphe **k-NN** sur la sphère céleste
- extraction de features (astro + métriques graphe avancées)
- détection d’anomalies (plusieurs engines + stratégies de seuil)
- explainability **LIME** + génération systématique de **prompts LLM**
- exports : CSV / GraphML / PNG / manifest JSON

## Exécution recommandée (sans “editable install”)

### Colab
Ouvrir `notebooks/colab_workflow.ipynb`.

### Local / Colab (repo root)
```bash
pip install -r requirements.txt
python run_workflow.py --mode csv --in-csv data/sample_gaia_like.csv --out results/run_csv --plots --explain-top 10
python run_workflow.py --mode gaia --ra 266.4051 --dec -28.936175 --radius-deg 0.5 --limit 2000 --out results/run_gaia --plots --explain-top 10
```

## Résultats
Chaque run produit :
- `raw.csv`, `scored.csv`, `top_anomalies.csv`
- `graph_full.graphml`, `graph_topk.graphml`
- `plots/*.png` (si `--plots`)
- `explanations.jsonl` + `llm_prompts.jsonl` (si `--explain-top > 0`)
- `manifest.json` (config + checksums)

## Engines disponibles
- `isolation_forest`
- `lof` (Local Outlier Factor)
- `ocsvm` (One-Class SVM)
- `robust_zscore`
- `pineforest` (optionnel : coniferest)

## Stratégies de seuil
- `contamination` (fraction)
- `percentile` (ex: 95)
- `top_k`
- `score` (seuil numérique sur anomaly_score)

## Métriques graphe (mode extended)
- degree, clustering
- k-core
- betweenness (approx possible)
- communautés (Louvain si dispo, sinon greedy modularity)
- articulation points
- bridges (projection node-level: incident_to_bridge)
