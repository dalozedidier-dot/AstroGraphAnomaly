from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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


def _first_existing_triplet(df: pd.DataFrame, candidates: Iterable[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    for a, b, c in candidates:
        if a in df.columns and b in df.columns and c in df.columns:
            return (a, b, c)
    return None


def _embedding_from_xyz_cols(df: pd.DataFrame) -> tuple[np.ndarray, str] | None:
    triplet = _first_existing_triplet(
        df,
        candidates=[
            ("x", "y", "z"),
            ("X", "Y", "Z"),
            ("cx", "cy", "cz"),
            ("px", "py", "pz"),
        ],
    )
    if triplet is None:
        return None

    a, b, c = triplet
    x = _to_float_series(df, a).to_numpy(dtype=float)
    y = _to_float_series(df, b).to_numpy(dtype=float)
    z = _to_float_series(df, c).to_numpy(dtype=float)
    xyz = np.vstack([x, y, z]).T
    return xyz, f"cols:{a},{b},{c}"


def _embedding_from_radec(df: pd.DataFrame) -> tuple[np.ndarray, str] | None:
    # We accept alternative spellings too
    ra_col = "ra" if "ra" in df.columns else ("ra_deg" if "ra_deg" in df.columns else None)
    dec_col = "dec" if "dec" in df.columns else ("dec_deg" if "dec_deg" in df.columns else None)
    if ra_col is None or dec_col is None:
        return None

    ra = _to_float_series(df, ra_col).to_numpy(dtype=float)
    dec = _to_float_series(df, dec_col).to_numpy(dtype=float)

    if not (np.isfinite(ra).any() and np.isfinite(dec).any()):
        return None

    base_xyz = _unit_sphere_xyz(ra, dec)

    # Optional bounded radial scaling from parallax (only when it is actually useful)
    parallax = _to_float_series(df, "parallax").to_numpy(dtype=float)
    parallax_err = _to_float_series(df, "parallax_error").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        snr = parallax / parallax_err

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
        dist_pc = 1000.0 / np.clip(parallax[use], 1e-3, np.inf)
        dist_pc = np.clip(dist_pc, 1.0, 20000.0)

        med = float(np.median(dist_pc)) if len(dist_pc) else 1.0
        if not np.isfinite(med) or med <= 0:
            med = 1.0

        r_use = dist_pc / med
        r[use] = r_use

    xyz = base_xyz * r[:, None]
    return xyz, f"radec:{ra_col},{dec_col}"


def _embedding_from_numeric_pca(df: pd.DataFrame) -> tuple[np.ndarray, str] | None:
    # Fallback when ra/dec are not present in the dataframe passed to build_knn_graph.
    # Build a stable embedding from numeric columns (after coercion), then PCA->3D.
    skip = {
        "source_id",
        "designation",
        "name",
        "classlabel_dsc",
        "in_vari_classification_result",
    }

    cols: list[str] = []
    for c in df.columns:
        if c in skip:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # Keep columns with enough signal
        if s.notna().mean() >= 0.5 and s.nunique(dropna=True) > 1:
            cols.append(c)

    if len(cols) < 2:
        return None

    X = np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in cols])

    imp = SimpleImputer(strategy="median")
    X = imp.fit_transform(X)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    n_comp = 3 if X.shape[1] >= 3 else 2
    pca = PCA(n_components=n_comp, random_state=0)
    E = pca.fit_transform(X)

    if E.shape[1] == 2:
        E = np.column_stack([E, np.zeros(len(df), dtype=float)])

    return E, f"pca:{len(cols)}cols"


def _compute_embedding(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    # Priority:
    # 1) explicit x/y/z columns (some pipelines precompute coords)
    # 2) ra/dec (always good)
    # 3) PCA over numeric columns (last resort)
    emb = _embedding_from_xyz_cols(df)
    if emb is not None:
        return emb

    emb = _embedding_from_radec(df)
    if emb is not None:
        return emb

    emb = _embedding_from_numeric_pca(df)
    if emb is not None:
        return emb

    raise ValueError(
        "No usable points to build kNN graph. "
        "Expected either ra/dec (or ra_deg/dec_deg), x/y/z, or >=2 usable numeric columns."
    )


def build_knn_graph(df: pd.DataFrame, knn_k: int = 8, include_self: bool = False) -> nx.Graph:
    """Build a kNN graph from a dataframe.

    This function is intentionally robust: it can build a graph from:
    - raw Gaia-like tables (ra/dec, optional parallax)
    - tables where a previous step already created x/y/z columns
    - feature tables where ra/dec are not present, via numeric PCA fallback

    This prevents CI failures on extragalactic samples (quasars/galaxies) where parallax is noisy
    and some pipelines drop too aggressively before kNN.
    """

    if knn_k < 1:
        raise ValueError("knn_k must be >= 1")

    xyz, mode = _compute_embedding(df)

    valid = np.isfinite(xyz).all(axis=1)
    if not np.any(valid):
        raise ValueError(
            "No finite rows for kNN after embedding. "
            "Check numeric coercion of embedding columns."
        )

    df_v = df.loc[valid].copy()
    xyz_v = xyz[valid]

    if "source_id" in df_v.columns:
        node_ids = df_v["source_id"].astype(str).to_list()
    else:
        node_ids = [str(i) for i in df_v.index.to_list()]

    n = len(node_ids)
    G = nx.Graph()
    G.graph["embedding_mode"] = mode

    if n == 1:
        _add_node_attrs(G, node_ids[0], df_v.iloc[0], xyz_v[0])
        return G

    n_neighbors = min(n, knn_k + (1 if not include_self else 0))
    if n_neighbors < 2:
        n_neighbors = 2

    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    nn.fit(xyz_v)
    dists, idxs = nn.kneighbors(xyz_v)

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

    for col in [
        "ra",
        "dec",
        "ra_deg",
        "dec_deg",
        "parallax",
        "parallax_error",
        "pmra",
        "pmra_error",
        "pmdec",
        "pmdec_error",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
        "classprob_dsc_combmod_quasar",
        "classprob_dsc_combmod_galaxy",
    ]:
        if col in row.index:
            try:
                val = float(row[col])
            except Exception:
                continue
            if np.isfinite(val):
                attrs[col] = val

    G.add_node(nid, **attrs)
