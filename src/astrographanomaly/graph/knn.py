from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def _to_float_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([np.nan] * len(df), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _unit_sphere_xyz(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    cosd = np.cos(dec)
    x = cosd * np.cos(ra)
    y = cosd * np.sin(ra)
    z = np.sin(dec)
    return np.vstack([x, y, z]).T


def build_knn_graph(df: pd.DataFrame, knn_k: int = 8, include_self: bool = False) -> nx.Graph:
    """Build a kNN graph from Gaia-like rows.

    Robust behavior:
    - Base embedding is always the unit celestial sphere (ra/dec -> 3D unit vector).
    - If parallax + parallax_error exist and are usable (positive and decent SNR),
      we apply a *bounded radial scaling* so nearby stars separate in 3D.
    - If parallax is missing or too noisy (common for quasars/galaxies), we keep r=1
      so the graph is still buildable and the 3D visualizations remain meaningful.

    Expected columns (minimum): ra, dec.
    Optional: parallax, parallax_error, pmra, pmdec, phot_g_mean_mag, bp_rp, ruwe, source_id.
    """

    if knn_k < 1:
        raise ValueError("knn_k must be >= 1")

    ra = _to_float_series(df, "ra").to_numpy(dtype=float)
    dec = _to_float_series(df, "dec").to_numpy(dtype=float)

    base_xyz = _unit_sphere_xyz(ra, dec)

    # Optional bounded radial scaling from parallax
    parallax = _to_float_series(df, "parallax").to_numpy(dtype=float)
    parallax_err = _to_float_series(df, "parallax_error").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = parallax / parallax_err

    # "Usable" parallax: positive, finite, error positive, SNR decent.
    # This is intentionally permissive: we only want to enrich when it helps,
    # not to delete the whole dataset.
    min_snr = 2.0
    use = (
        np.isfinite(parallax)
        & np.isfinite(parallax_err)
        & (parallax > 0.0)
        & (parallax_err > 0.0)
        & (snr >= min_snr)
    )

    r = np.ones(len(df), dtype=float)
    if np.any(use):
        # Convert parallax [mas] to distance [pc], and bound to avoid extreme leverage.
        # Gaia parallaxes can be tiny/noisy for extragalactic sources; bounding stabilizes.
        dist_pc = 1000.0 / np.clip(parallax[use], 1e-3, np.inf)
        dist_pc = np.clip(dist_pc, 1.0, 20000.0)

        med = float(np.median(dist_pc)) if len(dist_pc) else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0

        # Normalize around median so typical points stay near unit scale.
        r_use = dist_pc / med
        r[use] = r_use

    xyz = base_xyz * r[:, None]

    valid = np.isfinite(xyz).all(axis=1)
    if not np.any(valid):
        raise ValueError(
            "No usable points to build kNN graph. "
            "Check that ra/dec are present and numeric."
        )

    df_v = df.loc[valid].copy()
    xyz_v = xyz[valid]

    # Node ids
    if "source_id" in df_v.columns:
        node_ids = df_v["source_id"].astype(str).to_list()
    else:
        node_ids = [str(i) for i in df_v.index.to_list()]

    n = len(node_ids)
    if n == 1:
        G = nx.Graph()
        _add_node_attrs(G, node_ids[0], df_v.iloc[0], xyz_v[0])
        return G

    # sklearn returns self as nearest neighbor for metric spaces.
    n_neighbors = min(n, knn_k + (1 if not include_self else 0))
    if n_neighbors < 2:
        n_neighbors = 2

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(xyz_v)
    dists, idxs = nn.kneighbors(xyz_v)

    G = nx.Graph()
    for i, nid in enumerate(node_ids):
        _add_node_attrs(G, nid, df_v.iloc[i], xyz_v[i])

    for i, nid in enumerate(node_ids):
        for d, j in zip(dists[i], idxs[i], strict=False):
            if not include_self and j == i:
                continue
            if j == i:
                continue
            nid2 = node_ids[int(j)]
            if nid2 == nid:
                continue
            # Undirected graph: add each edge once
            if G.has_edge(nid, nid2):
                continue
            if not np.isfinite(d):
                continue
            G.add_edge(nid, nid2, weight=float(d))

    return G


def _add_node_attrs(G: nx.Graph, nid: str, row: pd.Series, xyz: np.ndarray) -> None:
    attrs: dict[str, Any] = {
        "x": float(xyz[0]),
        "y": float(xyz[1]),
        "z": float(xyz[2]),
    }
    # Keep common Gaia columns if present
    for col in [
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
    ]:
        if col in row.index:
            try:
                val = float(row[col])
            except Exception:
                continue
            if np.isfinite(val):
                attrs[col] = val

    G.add_node(nid, **attrs)
