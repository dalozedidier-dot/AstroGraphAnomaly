# Candidates 5p workflows v3

## Ce que tes logs montrent
Les runs "real_galaxy_candidates" et "real_quasar_candidates" utilisent encore une requete qui laisse passer des lignes avec
parallax, pm, ruwe manquants, et surtout sans les colonnes d'erreur. Le pipeline filtre ensuite a 0 ligne avant le kNN,
d'ou le crash NearestNeighbors.

## Ce pack
Ajoute deux nouveaux workflows avec des noms explicites et une requete Gaia DR3 qui garantit un noyau "5p astrometry" exploitable:
- real_galaxy_candidates_5p.yml
- real_quasar_candidates_5p.yml

Critères imposés:
- parallax > 0 + parallax_error non nul
- pmra, pmdec + leurs erreurs non nulles
- bp_rp, ruwe, phot_g_mean_mag non nuls
- ruwe <= ruwe_max

Chaque workflow affiche un sanity check (rows, rows with required 5p+phot).

## Utilisation
1) Copier les fichiers dans .github/workflows/
2) Commit + push
3) Dans Actions, lancer:
   - real_galaxy_candidates_5p
   - real_quasar_candidates_5p

Si tu gardes les anciens workflows, ne lance pas ceux la pour les candidats galaxies/quasars.
