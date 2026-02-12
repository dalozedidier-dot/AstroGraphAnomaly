# Real data workflow pack

Ce pack ajoute 5 workflows GitHub Actions pour exécuter AstroGraphAnomaly sur des données Gaia DR3 réelles via TAP public (ESA).

Fichiers

1) .github/workflows/real_gaia_cone_stars.yml
   Échantillon dans un cône RA/Dec (gaia_source), filtré par qualité (RUWE, magnitude) et avec parallaxe et mouvements propres.

2) .github/workflows/real_variability.yml
   Échantillon variabilité (vari_summary JOIN gaia_source). Option only_agn pour cibler les AGN.

3) .github/workflows/real_galaxy_candidates.yml
   Candidats galaxies (galaxy_candidates JOIN gaia_source) filtrés par probabilité galaxy.

4) .github/workflows/real_quasar_candidates.yml
   Quasar-like (galaxy_candidates JOIN gaia_source) filtrés par probabilité quasar.

5) .github/workflows/real_ruwe_outliers.yml
   Outliers astrométriques (gaia_source) via RUWE élevé.

Usage

1) Déposer les fichiers dans le repo.
2) Actions, choisir un workflow, Run workflow.
3) Les résultats sont uploadés en artefact sous results/<run_name>.

Notes

Ces workflows utilisent le TAP sync public. Si le service est lent ou indisponible, l’option fallback_to_sample permet d’utiliser data/sample_gaia_like.csv.
