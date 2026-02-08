#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaia smoke test robuste pour CI.

Objectif:
- tester l'accès TAP Gaia (astroquery.gaia) avec retry + fallback serveur
- extraire un petit CSV stable (TOP N, rayon réduit)
- optionnel: exécuter le pipeline AstroGraphAnomaly en mode CSV sur ce snapshot (end-to-end),
  sans dépendre du code fetch Gaia interne (qui reste utile en Colab, mais flakey en CI).

Paramétrage via env:
- GAIA_SMOKE_RA, GAIA_SMOKE_DEC, GAIA_SMOKE_RADIUS_DEG, GAIA_SMOKE_LIMIT
- GAIA_SMOKE_MAX_ATTEMPTS (default 4)
- GAIA_SMOKE_BACKOFF_BASE_S (default 2)
- GAIA_SMOKE_OUT (default results/gaia_smoke)
- GAIA_SMOKE_RUN_PIPELINE (0/1)
"""

from __future__ import annotations

import os
import sys
import time
import json
import random
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

# Heavy imports after basic env parsing (faster failure if deps missing)
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.gaia import Gaia

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _sleep_backoff(attempt: int, base: float) -> None:
    # backoff exponentiel + jitter
    delay = (base ** attempt) + random.uniform(0, 0.6)
    time.sleep(min(delay, 18.0))

def _looks_transient(msg: str) -> bool:
    m = msg.lower()
    # signatures usuelles sur runners CI
    return any(s in m for s in [
        "timeout", "timed out", "connectionreset", "connection reset",
        "broken pipe", "remote disconnected", "429", "too many requests",
        "temporary failure", "name resolution", "502", "503", "504",
        "ssl", "chunkedencodingerror"
    ])

def gaia_query_with_retry(query: str, max_attempts: int, backoff_base: float) -> pd.DataFrame:
    servers = [
        # ESAC (par défaut)
        "https://gea.esac.esa.int/tap-server/tap",
        # GAVO mirror (souvent plus stable selon la période)
        "https://gaia.ari.uni-heidelberg.de/tap",
    ]

    last_err: Optional[Exception] = None

    for server in servers:
        try:
            Gaia.MAIN_GAIA_SERVER = server  # type: ignore[attr-defined]
        except Exception:
            # si l'attribut n'existe pas (version), on ignore, astroquery utilisera son défaut
            pass

        for attempt in range(1, max_attempts + 1):
            try:
                job = Gaia.launch_job_async(query, verbose=False)
                table = job.get_results()
                df = table.to_pandas()
                if len(df) == 0:
                    raise RuntimeError("Empty result set (0 rows).")
                return df
            except Exception as e:
                last_err = e
                msg = f"{type(e).__name__}: {e}"
                print(f"[gaia_smoke] server={server} attempt={attempt}/{max_attempts} -> {msg}")

                # erreurs non transitoires -> stop immédiat sur ce serveur
                if not _looks_transient(str(e)):
                    break

                if attempt < max_attempts:
                    _sleep_backoff(attempt, backoff_base)

        # next server fallback

    if last_err is None:
        raise RuntimeError("Gaia query failed: unknown error.")
    raise last_err

def main() -> int:
    ra = _env_float("GAIA_SMOKE_RA", 266.4)
    dec = _env_float("GAIA_SMOKE_DEC", -29.0)
    radius_deg = _env_float("GAIA_SMOKE_RADIUS_DEG", 0.15)
    limit = _env_int("GAIA_SMOKE_LIMIT", 30)
    max_attempts = _env_int("GAIA_SMOKE_MAX_ATTEMPTS", 4)
    backoff_base = _env_float("GAIA_SMOKE_BACKOFF_BASE_S", 2.0)
    out_dir = Path(os.environ.get("GAIA_SMOKE_OUT", "results/gaia_smoke"))
    run_pipeline = os.environ.get("GAIA_SMOKE_RUN_PIPELINE", "1").strip() == "1"

    out_dir.mkdir(parents=True, exist_ok=True)

    # Versions
    try:
        import astroquery
        astroquery_version = getattr(astroquery, "__version__", "unknown")
    except Exception:
        astroquery_version = "unknown"

    meta = {
        "ra": ra, "dec": dec, "radius_deg": radius_deg, "limit": limit,
        "max_attempts": max_attempts, "backoff_base": backoff_base,
        "astroquery_version": astroquery_version,
        "python": sys.version,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[gaia_smoke] meta:", json.dumps(meta, indent=2))

    # ADQL (ICRS explicite)
    query = f"""
    SELECT TOP {limit}
        source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    ) = 1
    AND parallax IS NOT NULL AND parallax > 0
    """.strip()

    t0 = time.time()
    df = gaia_query_with_retry(query, max_attempts=max_attempts, backoff_base=backoff_base)
    dt = time.time() - t0

    # distance approx
    df["distance"] = 1000.0 / df["parallax"]
    df = df.dropna()

    raw_csv = out_dir / "gaia_smoke_raw.csv"
    df.to_csv(raw_csv, index=False)
    print(f"[gaia_smoke] OK rows={len(df)} dt={dt:.1f}s -> {raw_csv}")

    if not run_pipeline:
        return 0

    # Pipeline end-to-end via run_workflow.py en mode csv
    # => si run_workflow.py n'existe pas, on stop avec erreur explicite
    if not Path("run_workflow.py").exists():
        print("[gaia_smoke] ERROR: run_workflow.py missing (cannot run pipeline)")
        return 1

    out_run = out_dir / "pipeline_out"
    out_run.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "run_workflow.py",
        "--mode", "csv",
        "--in-csv", str(raw_csv),
        "--out", str(out_run),
        "--plots",
        "--explain-top", "3",
        "--top-k", "10",
    ]

    print("[gaia_smoke] pipeline:", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[gaia_smoke] pipeline failed rc={rc}")
        return rc

    print("[gaia_smoke] pipeline OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
