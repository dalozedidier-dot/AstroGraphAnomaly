# AstroGraphAnomaly — Fix Matplotlib canvas for Viz A→H (3D)

## Problème
Le workflow `manual_viz_a_to_h` plante pendant la génération du GIF `07_proper_motion_trails.gif` avec:

```
AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
```

C’est un changement récent de Matplotlib: `FigureCanvasAgg.tostring_rgb()` n’est plus disponible.

## Solution
Ce zip contient un script qui patche automatiquement `tools/viz_a_to_h_suite.py` pour utiliser:
- `fig.canvas.buffer_rgba()`
- reshape RGBA → RGB

## Application (local ou GitHub Codespaces / github.dev)
Depuis la racine du repo:

```bash
python scripts/fix_viz_a_to_h_matplotlib_canvas.py --file tools/viz_a_to_h_suite.py
```

Le script:
- écrit un backup `tools/viz_a_to_h_suite.py.bak`
- remplace la capture d’image basée sur `tostring_rgb()`

Ensuite:

```bash
git add tools/viz_a_to_h_suite.py tools/viz_a_to_h_suite.py.bak || true
git commit -m "fix: Matplotlib FigureCanvasAgg tostring_rgb -> buffer_rgba for GIF"
git push
```

Relance ton workflow `manual_viz_a_to_h`.

## Notes
- Le 3D est produit sous forme de fichiers `.html` (Plotly/PyVis) dans `results/<run>/viz_a_to_h/`.
- Si un autre `tostring_rgb` existe ailleurs dans le fichier, le script te prévient.
