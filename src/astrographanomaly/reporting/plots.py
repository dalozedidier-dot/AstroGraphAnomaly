from __future__ import annotations

from pathlib import Path
import math
import networkx as nx
import matplotlib.pyplot as plt


def save_graph_plot(out_dir: str | Path, G: nx.Graph, anomalies: set) -> None:
    """
    Plot du graphe en garantissant une position pour chaque nœud.
    - utilise node["pos"] si présent
    - sinon utilise (ra, dec) si présents
    - sinon fallback spring_layout pour les nœuds manquants
    - ajoute aussi les clés str(node) pour éviter mismatch int/str
    """
    out_dir = Path(out_dir)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    pos: dict = {}

    def _is_finite(x) -> bool:
        try:
            return x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))
        except Exception:
            return False

    def _add_pos(k, xy):
        # key direct
        pos[k] = xy
        # key str fallback (si graph nodes sont des strings)
        try:
            pos[str(k)] = xy
        except Exception:
            pass

    # 1) Positions issues des attributs
    for n in G.nodes():
        data = G.nodes[n]

        # a) pos explicite
        if "pos" in data:
            p = data.get("pos")
            if isinstance(p, (tuple, list)) and len(p) == 2 and _is_finite(p[0]) and _is_finite(p[1]):
                _add_pos(n, (float(p[0]), float(p[1])))
                continue

        # b) ra/dec
        ra = data.get("ra", None)
        dec = data.get("dec", None)
        if _is_finite(ra) and _is_finite(dec):
            _add_pos(n, (float(ra), float(dec)))

    # 2) Compléter les positions manquantes (fallback layout)
    missing = [n for n in G.nodes() if n not in pos and str(n) not in pos]
    if missing:
        # layout pour tout le graphe, puis overwrite si positions existantes
        spring = nx.spring_layout(G, seed=42)
        for n in missing:
            if n in spring:
                _add_pos(n, spring[n])

    # 3) Sécurité finale : si toujours incomplet -> spring_layout total
    if any((n not in pos and str(n) not in pos) for n in G.nodes()):
        spring = nx.spring_layout(G, seed=42)
        pos = {}
        for n, xy in spring.items():
            _add_pos(n, xy)

    node_colors = ["red" if (n in anomalies or str(n) in anomalies) else "blue" for n in G.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, node_color=node_colors, node_size=20, with_labels=False, alpha=0.85, width=0.4)
    plt.title("Graph (anomalies en rouge)")
    plt.tight_layout()
    plt.savefig(plot_dir / "graph_anomalies.png", dpi=160)
    plt.close()
