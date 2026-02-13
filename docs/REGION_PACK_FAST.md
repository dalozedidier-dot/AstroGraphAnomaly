# Region Pack fast

Ce module sert à scorer rapidement des fichiers Gaia DR3 issus de tables dérivées, typiquement :

- `GalaxyCandidates_*.csv.gz` (table `galaxy_candidates`)
- `VariSummary_*.csv.gz` (table `vari_summary`)
- optionnel : `GalaxyCatalogueName_*.csv.gz` (mapping `catalogue_id` par `source_id`)

Contraintes
- Ces tables ne contiennent généralement pas `ra/dec`, donc elles ne passent pas directement par le pipeline sky-graph principal.
- Le script produit des artefacts tabulaires et une projection PCA 2D, sans sky map.

## Usage

### Exécuter sur des fichiers

```bash
python tools/region_pack_fast.py   --kind galaxy_candidates   --inputs data/region_pack/raw/GalaxyCandidates_*.csv.gz   --out results/region_pack_fast/galaxy_candidates   --engine robust_zscore   --top-k 200   --max-rows 0
```

### Smoke CI

Le workflow GitHub Actions `region_pack_fast_smoke.yml` exécute :

```bash
python tools/ci_region_pack_fast_smoke.py --max-rows 6000 --top-k 120
```

Les artefacts sont uploadés dans `results/region_pack_fast_ci/`.

## Sorties

Pour chaque région `RID` :

- `results/.../region_<RID>/scored.csv.gz`
- `results/.../region_<RID>/01_embedding_pca.png`
- `results/.../region_<RID>/summary.json`
- `results/.../region_<RID>/index.html`

Et un agrégat :

- `results/.../<kind>_scored_all.csv.gz`
