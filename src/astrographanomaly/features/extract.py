from __future__ import annotations
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional

def _community_ids(G: nx.Graph, method: str = "louvain") -> Dict[int, int]:
    """Return node->community_id mapping with graceful fallback."""
    communities = None
    if method == "louvain":
        try:
            from networkx.algorithms.community import louvain_communities
            communities = louvain_communities(G, seed=42)
        except Exception:
            communities = None

    if communities is None:
        from networkx.algorithms.community import greedy_modularity_communities
        communities = greedy_modularity_communities(G)

    mapping = {}
    for cid, comm in enumerate(communities):
        for n in comm:
            mapping[int(n)] = cid
    return mapping

def extract_features(
    G: nx.Graph,
    mode: str = "extended",
    betweenness_enabled: bool = True,
    betweenness_k: Optional[int] = 300,
    betweenness_seed: int = 42,
    communities_method: str = "louvain",
    articulation_points: bool = True,
    bridges: bool = True,
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Node feature matrix from graph + node attributes.

    Always includes: degree, clustering, parallax, pmra, pmdec, mag, distance.
    Extended adds: kcore, betweenness, community_id, is_articulation, incident_to_bridge.
    """
    nodes = list(G.nodes())
    n = len(nodes)

    deg = dict(G.degree())
    clust = nx.clustering(G)

    # Base astro attributes
    parallax = {u: G.nodes[u].get("parallax", np.nan) for u in nodes}
    pmra = {u: G.nodes[u].get("pmra", np.nan) for u in nodes}
    pmdec = {u: G.nodes[u].get("pmdec", np.nan) for u in nodes}
    mag = {u: G.nodes[u].get("mag", np.nan) for u in nodes}
    dist = {u: G.nodes[u].get("distance", np.nan) for u in nodes}

    feature_names = ["degree","clustering","parallax","pmra","pmdec","mag","distance"]

    extra = {}
    if mode == "extended":
        # k-core
        core = nx.core_number(G) if n > 0 else {}
        extra["kcore"] = {u: core.get(u, 0) for u in nodes}
        feature_names.append("kcore")

        # betweenness (optionally approximate)
        if betweenness_enabled and n > 1:
            if betweenness_k is None:
                btw = nx.betweenness_centrality(G, normalized=True)
            else:
                # approximate with k node samples
                btw = nx.betweenness_centrality(G, k=min(int(betweenness_k), n), normalized=True, seed=int(betweenness_seed))
            extra["betweenness"] = {u: btw.get(u, 0.0) for u in nodes}
        else:
            extra["betweenness"] = {u: 0.0 for u in nodes}
        feature_names.append("betweenness")

        # communities
        comm = _community_ids(G, method=communities_method) if n > 0 else {}
        extra["community"] = {u: comm.get(int(u), -1) for u in nodes}
        feature_names.append("community")

        # articulation points
        if articulation_points and n > 2:
            aps = set(nx.articulation_points(G))
        else:
            aps = set()
        extra["is_articulation"] = {u: (1 if u in aps else 0) for u in nodes}
        feature_names.append("is_articulation")

        # bridges (node-level: incident to at least one bridge edge)
        if bridges and n > 1 and G.number_of_edges() > 0:
            try:
                br = list(nx.bridges(G))
                bridge_nodes = set([a for a,b in br] + [b for a,b in br])
            except Exception:
                bridge_nodes = set()
        else:
            bridge_nodes = set()
        extra["incident_to_bridge"] = {u: (1 if u in bridge_nodes else 0) for u in nodes}
        feature_names.append("incident_to_bridge")

    rows = []
    for u in nodes:
        row = [
            deg.get(u, 0),
            clust.get(u, 0.0),
            parallax.get(u, np.nan),
            pmra.get(u, np.nan),
            pmdec.get(u, np.nan),
            mag.get(u, np.nan),
            dist.get(u, np.nan),
        ]
        if mode == "extended":
            row += [
                extra["kcore"].get(u, 0),
                extra["betweenness"].get(u, 0.0),
                extra["community"].get(u, -1),
                extra["is_articulation"].get(u, 0),
                extra["incident_to_bridge"].get(u, 0),
            ]
        rows.append(row)

    X = np.asarray(rows, dtype=float)
    return X, [int(u) for u in nodes], feature_names
