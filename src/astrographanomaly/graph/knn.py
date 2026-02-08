from __future__ import annotations
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def radec_to_unit_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return np.column_stack([x,y,z])

def build_knn_graph(df, knn_k: int = 8, include_self: bool = False) -> nx.Graph:
    """Build an undirected k-NN graph in 3D unit-vector space (sky sphere)."""
    node_ids = df["source_id"].to_numpy()
    xyz = radec_to_unit_xyz(df["ra"].to_numpy(), df["dec"].to_numpy())

    # n_neighbors: add 1 if include_self=False to avoid self neighbor in results
    n_neighbors = knn_k + (0 if include_self else 1)
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(xyz)
    dists, idxs = nn.kneighbors(xyz, return_distance=True)

    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(int(row["source_id"]),
                   ra=float(row["ra"]), dec=float(row["dec"]),
                   parallax=float(row.get("parallax", np.nan)) if "parallax" in row else np.nan,
                   pmra=float(row.get("pmra", np.nan)) if "pmra" in row else np.nan,
                   pmdec=float(row.get("pmdec", np.nan)) if "pmdec" in row else np.nan,
                   mag=float(row.get("phot_g_mean_mag", row.get("mag", np.nan))) if ("phot_g_mean_mag" in row or "mag" in row) else np.nan,
                   distance=float(row.get("distance", np.nan)) if "distance" in row else np.nan)

    for i, neigh in enumerate(idxs):
        for j in neigh:
            if not include_self and j == i:
                continue
            a = int(node_ids[i])
            b = int(node_ids[j])
            if a != b:
                G.add_edge(a, b)

    return G
