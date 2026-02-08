Patch "dual option" — sans choix par défaut
==========================================

Tu ne veux pas choisir entre :
(A) intégration "dans le pipeline"
(B) post-step séparé

Ce patch ne touche pas au cœur et fournit 2 chemins, co-existants :

Option B (post-step)
--------------------
Après un run qui produit `scored.csv` (+ `graph_full.graphml`):
python tools/full_plots_suite.py --scored results/<run>/scored.csv --graph results/<run>/graph_full.graphml --out results/<run>/plots --top-k 30 --write-enriched

Wrapper (1 commande)
--------------------
python tools/run_full.py csv --in-csv data/sample_gaia_like.csv --out results/run_full --top-k 30 --knn-k 8 --engine isolation_forest

CI
--
Un workflow `ci_full_artifacts.yml` est fourni : il exécute `tools/run_full.py` et vérifie >= 8 PNG.
