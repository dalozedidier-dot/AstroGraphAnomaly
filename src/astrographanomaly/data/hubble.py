"""Hubble/HST ingestion support (optional).

This repository supports a *Hubble-like* catalog ingestion path to keep the workflow consistent.
Live MAST queries are intentionally not enabled by default due to schema/availability variability.

Expected minimal columns after normalization: source_id, ra, dec.
Optional: mag, flux, parallax, pmra, pmdec, distance.
"""

import pandas as pd

def load_hubble_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalize coordinates
    rename = {}
    if "RA" in df.columns and "ra" not in df.columns:
        rename["RA"] = "ra"
    if "DEC" in df.columns and "dec" not in df.columns:
        rename["DEC"] = "dec"
    df = df.rename(columns=rename)

    # Normalize id
    if "source_id" not in df.columns:
        if "objID" in df.columns:
            df = df.rename(columns={"objID": "source_id"})
        else:
            df["source_id"] = range(1, len(df) + 1)

    return df.dropna(subset=["ra", "dec"])
