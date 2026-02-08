# Patch: Matplotlib colorbar hotfix (CI full plots)

This patch adds a `sitecustomize.py` at repository root.

It prevents CI failures in `tools/full_plots_suite.py` where Matplotlib may raise
when calling `plt.colorbar(...)` without an explicit `ax=`.

It only changes behavior for the specific ValueError message:
"Unable to determine Axes to steal space for Colorbar..."

Opt-out:
- set `ASTROGRAPHANOMALY_DISABLE_SITECUSTOMIZE=1`

Install:
- Drop the file at repository root (same level as `pyproject.toml`)
