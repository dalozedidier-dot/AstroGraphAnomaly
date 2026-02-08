Patch gaia_smoke (robust)
========================

Objectif
--------
Réduire la variabilité de `gaia_smoke` (60% failure rate observé dans les GitHub Insights)
en traitant explicitement:
- instabilité réseau/timeout/reset sur TAP
- 429 (rate limiting) via retry/backoff
- fallback serveur TAP (ESAC -> GAVO)

Ce patch remplace/ajoute
------------------------
- .github/workflows/gaia_smoke.yml
- tools/gaia_smoke_test.py

Comportement
------------
- workflow déclenché uniquement via:
  - workflow_dispatch
  - schedule (quotidien)
- python 3.11 (stabilité en CI)
- artefacts uploadés dans actions: results/gaia_smoke/*

Notes
-----
- Le smoke test fetch un snapshot Gaia minimal, puis exécute le pipeline en mode CSV.
  Cela isole la flakiness Gaia du reste du pipeline.
- Pour désactiver le pipeline end-to-end, mettre GAIA_SMOKE_RUN_PIPELINE=0 dans le workflow.
