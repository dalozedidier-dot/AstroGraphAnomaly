Correctifs Viz A à H

Ce zip contient une version complète de tools/viz_a_to_h_suite.py corrigée pour deux erreurs CI :

1) Matplotlib: FigureCanvasAgg n'a plus tostring_rgb()
   Fix: capture image via canvas.buffer_rgba().

2) Pandas/NumPy: ValueError "assignment destination is read-only" dans export_umap()
   Fix: to_numpy(copy=True) + garde-fou si ndarray read-only (Copy-on-Write).

Application
1) Dézipper à la racine du repo (écrase tools/viz_a_to_h_suite.py)
2) Commit + push
3) Relancer le workflow manual_viz_a_to_h
