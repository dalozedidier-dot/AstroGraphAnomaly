# Colab Image Report Notebook (2026)

This patch adds a Colab notebook that generates the same kind of PNG plots you shared
(ra/dec maps, CMD, score histograms, top anomalies, k-NN graph views, communities).

## Quick usage

1. Run your AstroGraphAnomaly pipeline so you have a run folder like:

`results/<run>/raw.csv`
`results/<run>/scored.csv`
`results/<run>/top_anomalies.csv`
`results/<run>/graph_full.graphml` (optional)
`results/<run>/scored_enriched.csv` (optional)

2. Open:

`notebooks/colab_image_report_2026.ipynb`

3. Set `RUN_DIR` to your run folder and execute the notebook.

Outputs are written to:

`<run-dir>/image_report/`

including `index.html` and `report.json`.

## CLI alternative

You can also generate the report without Colab:

```bash
python tools/generate_image_report.py --run-dir results/<run>
```
