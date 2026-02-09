# Colab Analysis to Images Report (2026)

This notebook recomputes the analyses and generates the figure set (graph views, RA/Dec, mean features, top anomalies, CMD, region distribution, communities).

## How to use
Recommended:
1) Upload a CI artifact ZIP (for example `ci_full_py3.12.zip`).
2) The notebook extracts it and auto-detects the folder that contains `scored.csv`.

Advanced:
- Mount Google Drive and set `RUN_ROOT` to a folder that contains your `results/<run>/...` directory.
- Or set `FORCE_RUN_DIR` directly if you already know the exact folder.

## Inputs expected (best case)
- scored.csv (required)
- raw.csv (optional but enables region distribution plot)
- top_anomalies.csv (optional, computed if missing)
- graph_full.graphml (optional, computed kNN graph if missing)
- scored_enriched.csv (optional, used for community_id if present)

## Output
The notebook writes:
- <RUN_DIR>/image_report/*.png
- <RUN_DIR>/image_report/index.html
And it offers a ZIP download: astrograph_image_report.zip
