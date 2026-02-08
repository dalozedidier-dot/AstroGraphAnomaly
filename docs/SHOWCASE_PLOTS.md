Showcase plots suite
===================

But
---
Produire une sortie visuelle "au top" (plots cohérents + nombreux) à partir de `scored.csv`,
sans dépendre d'un packaging.

Commande
--------
python tools/showcase_plots.py \
  --scored results/<run>/scored.csv \
  --graph  results/<run>/graph_full.graphml \
  --out    results/<run> \
  --top-k  30 \
  --write-enriched \
  --copy-to-screenshots screenshots

Sorties
-------
- results/<run>/plots/*.png (≈18)
- results/<run>/scored_enriched.csv (optionnel)
- results/<run>/graph_metrics.json (optionnel)
- screenshots/*.png (6 curated, optionnel)
