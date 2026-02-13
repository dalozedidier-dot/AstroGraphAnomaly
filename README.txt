AstroGraphAnomaly Region Pack (fast)

Généré à partir de fichiers Gaia DR3:
- GalaxyCandidates_*.csv.gz (table galaxy_candidates)
- VariSummary_*.csv.gz (table vari_summary)
- GalaxyCatalogueName_*.csv.gz (mapping catalogue_id pour une partie des sources)

Sorties
- results/<kind>/region_<RID>/scored.csv.gz
- results/<kind>/region_<RID>/01_embedding_pca.png
- results/<kind>/region_<RID>/summary.json
- results/<kind>/region_<RID>/index.html

Limite
- Pas de ra/dec ici, donc pas de sky map. Pour enrichir: tools/fetch_gaia_source_tap.py puis jointure sur source_id.
