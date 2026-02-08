AstroGraphAnomaly patch bundle
=============================

Contenu (fichiers à remplacer/ajouter dans ton repo) :

1) notebooks/colab_workflow.ipynb (REMPLACER)
2) notebooks/colab_gaia_run.ipynb (AJOUTER)
3) notebooks/colab_inspect_outputs.ipynb (AJOUTER)

4) src/astrographanomaly/data/gaia.py (REMPLACER)
   - Ajoute bp_rp dans la requête Gaia (utile pour CMD)

5) src/astrographanomaly/reporting/plots.py (REMPLACER)
   - Ajoute cmd_bp_rp_vs_g.png si bp_rp disponible
   - Rend save_graph_plot robuste (plus de crash "Node has no position")

Procédure GitHub web :
- Ouvrir chaque fichier cible → Edit → coller → Commit
OU
- Dézipper localement puis upload les fichiers aux mêmes chemins.

