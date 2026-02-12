# v4 workflows: avoid all-NULL columns that can empty the dataset

## Symptom (your run 57223082771)
- Fetch step: 1554 rows downloaded, looks fine.
- Pipeline step: crashes in build_knn_graph with "No usable points ...".

This is consistent with a global `dropna()` (or equivalent) happening in the pipeline:
if the CSV contains a column that is NULL for all rows, `dropna()` drops every row -> df becomes empty -> kNN can't build.

In your quasar sample, `redshift_ugc` is commonly NULL for all rows, so keeping it in the SELECT can trigger this behavior.

## Fix in v4
- Remove `redshift_ugc` from the quasar/galaxy SELECTs.
- Add a "Sanitize CSV" step that drops any column that is 100% NaN and prints remaining NaN columns.

This makes the workflow robust even if the pipeline does global dropna.

## Installation
Replace the two workflow files in your repo:

- .github/workflows/real_quasar_candidates_5p.yml
- .github/workflows/real_galaxy_candidates_5p.yml

Commit + push, then rerun the workflows.
