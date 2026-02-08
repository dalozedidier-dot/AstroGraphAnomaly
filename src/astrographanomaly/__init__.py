"""AstroGraphAnomaly

Few-label anomaly detection on astronomical catalogs using graph-based features.
Primary target: Gaia DR3 sources. Optional catalog ingestion for Hubble/HST via MAST (astroquery),
depending on availability.

This package is workflow-first (CLI + pipeline), optimized for GitHub-web + Colab usage.
"""

__all__ = ["pipeline"]
__version__ = "0.2.0"
