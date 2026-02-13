Region Pack (fast)
================

But
---
Traiter rapidement des chunks Gaia DR3 (ex: `GalaxyCandidates_*.csv.gz`, `VariSummary_*.csv.gz`) pour produire
des sorties par "région" (chunk) avec:
- un score d'anomalie simple basé sur distance kNN
- un embedding PCA2
- un mini dashboard HTML par région

Scripts
-------
- `tools/run_regions_fast.py`: runner principal (chunk → scored + PCA + index.html)
- `tools/fetch_gaia_source_tap.py`: enrichissement TAP sur `source_id` pour récupérer `ra/dec/...`
- `tools/split_by_sky_tiles.py`: découpe un CSV enrichi en tuiles RA/Dec (création de nouvelles régions)

Dépendances
-----------
- `jinja2` (génération HTML)
- `requests` (TAP)
Ces deps sont incluses dans `requirements.txt`.

Exemples
--------
```bash
python tools/run_regions_fast.py --kind galaxy_candidates --inputs "data/GalaxyCandidates_*.csv.gz" --out results/galaxy_candidates_fast
python tools/run_regions_fast.py --kind vari_summary     --inputs "data/VariSummary_*.csv.gz"     --out results/vari_summary_fast
```

Limite
------
Les tables `galaxy_candidates` / `vari_summary` ne contiennent pas `ra/dec`. Pour obtenir les sky maps:
1) extraire des `source_id`
2) enrichir via `fetch_gaia_source_tap.py`
3) joindre `ra/dec` dans `scored.csv`
