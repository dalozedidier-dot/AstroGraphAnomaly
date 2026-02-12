# Patch v2: kNN multi-fallback embedding

## Ce que montrent tes logs (run 57219614115)
Le dataset téléchargé est OK (1554 lignes, ra/dec présents), mais `build_knn_graph` plante sur:

ValueError: No usable points to build kNN graph. Check that ra/dec are present and numeric.

Ça indique que le dataframe passé au builder n'a plus ra/dec (ou qu'il a été transformé en features),
donc l'embedding ne peut pas se faire.

## Ce patch
Remplace `src/astrographanomaly/graph/knn.py` par une version robuste avec 3 niveaux de fallback:

1) Si `x,y,z` existent déjà => on les utilise directement
2) Sinon si `ra/dec` (ou `ra_deg/dec_deg`) existent => sphère céleste + radial scaling optionnel
3) Sinon => PCA sur colonnes numériques (>=2 colonnes utilisables), puis embedding 3D

## Résultat
Plus de crash "0 sample(s)" ou "No usable points" sur les workflows quasar/galaxy.
Le graphe kNN est toujours construit et la suite A à H peut tourner.

## Installation
Déposer ce fichier en remplacement:

src/astrographanomaly/graph/knn.py

Puis relancer le workflow.
