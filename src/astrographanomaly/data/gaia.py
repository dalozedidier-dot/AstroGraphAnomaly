from __future__ import annotations
import pandas as pd

def fetch_gaia(ra: float, dec: float, radius_deg: float, limit: int) -> pd.DataFrame:
    """Fetch Gaia DR3 subset via ADQL (requires network + astroquery)."""
    from astroquery.gaia import Gaia

    query = f"""
    SELECT TOP {limit}
      source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE 1=CONTAINS(POINT(ra, dec), CIRCLE({ra}, {dec}, {radius_deg}))
      AND parallax IS NOT NULL AND parallax > 0
    """
    job = Gaia.launch_job(query)
    table = job.get_results()
    df = table.to_pandas()
    df["distance"] = 1000.0 / df["parallax"]
    return df.dropna()
