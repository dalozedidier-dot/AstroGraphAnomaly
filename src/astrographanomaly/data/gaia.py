from __future__ import annotations

import pandas as pd

def fetch_gaia(ra: float, dec: float, radius_deg: float, limit: int) -> pd.DataFrame:
    """Fetch Gaia DR3 sources around (ra, dec) within radius_deg.

    Adds:
      - distance = 1000/parallax (pc, approx)
    Includes:
      - bp_rp (for CMD plots) if available in Gaia source table
    """
    from astroquery.gaia import Gaia

    query = f"""
    SELECT TOP {limit}
      source_id, ra, dec,
      parallax, pmra, pmdec,
      phot_g_mean_mag,
      bp_rp
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_deg}))
      AND parallax IS NOT NULL AND parallax > 0
    """

    job = Gaia.launch_job(query)
    table = job.get_results()
    df = table.to_pandas()

    # distance (pc) from parallax (mas), ignoring uncertainties
    if "parallax" in df.columns:
        df["distance"] = 1000.0 / df["parallax"]

    # keep minimal required
    df = df.dropna(subset=["source_id", "ra", "dec"])
    return df.dropna()
