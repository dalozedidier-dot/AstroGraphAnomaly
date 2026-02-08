# AstroGraphAnomaly
Graph-based anomaly detection for astronomical catalog data (**Gaia** + **CSV**).  
Pipeline : k-NN graph construction (sky) → graph features → multi-detectors → explainability (LIME) → artefacts (CSV/GraphML/plots) + prompts LLM.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dalozedidier-dot/AstroGraphAnomaly/blob/main/notebooks/colab_workflow.ipynb)
![CI](https://img.shields.io/github/actions/workflow/status/dalozedidier-dot/AstroGraphAnomaly/ci.yml?label=CI)

> Note CI: les workflows "Gaia" sont volontairement séparés (réseau/quota instables).  
> Le pipeline offline (CSV) sert de test reproductible.

## Features
- **Modes** : `csv` (offline) et `gaia` (astroquery TAP)  
- **Graph** : k-NN (sky) + métriques (degree, clustering, k-core, betweenness approx, communautés)  
- **Detectors** : IsolationForest / LOF / OC-SVM / robust z-score (selon config)  
- **Explainability** : LIME (top anomalies) + génération de prompts LLM (JSONL)  
- **Artefacts** : `raw.csv`, `scored.csv`, `top_anomalies.csv`, `graph_full.graphml`, `graph_topk.graphml`, `manifest.json`, `plots/*.png`

## Quick start (Colab)
1. Ouvrir le notebook Colab ci-dessus.
2. Lancer le workflow offline (CSV) puis optionnellement Gaia.

## Quick start (CLI workflow-first)
### CSV
```bash
python workflow.py csv   --in-csv data/sample_gaia_like.csv   --out results/run_csv   --top-k 30 --knn-k 8   --plots --explain-top 5   --features-mode extended
```

### Gaia (optionnel, réseau)
```bash
python workflow.py gaia   --ra 266.4051 --dec -28.936175 --radius-deg 0.3 --limit 1200   --out results/run_gaia   --top-k 30 --knn-k 8   --plots --explain-top 5   --features-mode extended
```

## Outputs (structure)
Exemple :
```
results/run_csv/
  raw.csv
  scored.csv
  top_anomalies.csv
  graph_full.graphml
  graph_topk.graphml
  manifest.json
  explanations.jsonl
  llm_prompts.jsonl
  plots/
    score_hist.png
    ra_dec_score.png
    mag_vs_distance.png
    top_anomalies_scores.png
    pca_2d.png
    graph_communities_anomalies.png
    cmd_bp_rp_vs_g.png        # si bp_rp disponible (Gaia ou CSV enrichi)
```

## Screenshots
Ajouter 3–8 PNG dans `screenshots/` et référencer ici :
- `screenshots/anomaly_scores.png`
- `screenshots/graph_communities.png`
- `screenshots/cmd.png`

## Contributing
Voir `CONTRIBUTING.md`.

## Citation
Voir `CITATION.cff`.
