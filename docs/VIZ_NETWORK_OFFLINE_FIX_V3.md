# Patch v3: Network explorer HTML 100% offline

## Symptôme
Le fichier `viz_a_to_h/03_network_explorer.html` référence des scripts externes, typiquement:
- `../node_modules/vis/dist/vis.js` (inexistant dans les artefacts)
- bootstrap via CDN

Du coup, l'explorateur réseau s'ouvre mais reste vide ou cassé.

## Pourquoi
Selon la version de PyVis installée, l'option `cdn_resources="in_line"` peut ne pas exister.
Le script tombait alors en fallback PyVis "local", qui génère du HTML dépendant de `node_modules` ou d'un CDN.

## Fix
- PyVis est utilisé uniquement si le HTML généré est réellement offline (pas de node_modules, pas de CDN, pas de script src externe)
- Sinon, fallback automatique sur un explorateur réseau Plotly (Plotly inline), totalement autonome

## Installation
Remplacer le fichier:
- `tools/viz_a_to_h_suite.py`

Puis relancer un run et regénérer la suite A→H.
