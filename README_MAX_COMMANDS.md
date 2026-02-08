AstroGraphAnomaly — Max commands (workflow-first)
=================================================

But
---
Exécuter le workflow avec une couverture maximale (engines × seuils) en offline,
option Gaia (RUN_GAIA=1), puis compléter avec une post-analyse (métriques graphe + plots diagnostics)
et vérifier le contrat d'artefacts.

Fichiers à ajouter au repo
--------------------------
- tools/run_max_all.py
- tools/post_analysis.py
- tools/validate_outputs.py
- tools/__init__.py

Exécution
---------
Offline:
    python tools/run_max_all.py

Gaia (optionnel):
    RUN_GAIA=1 AGA_LIMIT=1200 python tools/run_max_all.py

Variables utiles
----------------
- AGA_TOPK=50
- AGA_EXPLAIN_TOP=15
- AGA_KNNK=10
- RUN_GAIA=1
- AGA_RA=266.4051
- AGA_DEC=-28.936175
- AGA_RADIUS=0.3
- AGA_LIMIT=1200

Sorties
-------
- results/max_runs/max_runs_summary.json
- results/max_runs/<run>/... (+ analysis/ dans chaque run OK)
