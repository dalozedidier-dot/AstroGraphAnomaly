# Patch: kNN graph fallback for extragalactic candidates

## Symptôme
Sur des datasets "quasar_candidates" ou "galaxy_candidates", le run crash dans:

ValueError: Found array with 0 sample(s) (shape=(0, 3)) while a minimum of 1 is required by NearestNeighbors.

## Pourquoi
Les candidats extragalactiques ont souvent des parallaxes et erreurs très bruitées, parfois quasi nulles.
L'ancien build_knn_graph fabriquait un embedding 3D basé sur la parallaxe et finissait par dropper toutes les lignes,
d'où xyz vide.

## Fix
- Embedding de base = sphère céleste (ra/dec -> vecteur 3D unitaire) pour tous les objets
- Parallaxe utilisée seulement si elle est positive et avec un SNR minimal (>= 2), avec une échelle radiale bornée
- Plus aucun cas où l'intégralité du dataset saute avant le kNN

## Installation
Déposer ce fichier en remplacement:

src/astrographanomaly/graph/knn.py

Puis relancer le workflow.
