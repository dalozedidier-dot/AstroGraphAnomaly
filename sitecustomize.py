"""
sitecustomize.py

Hotfix for Matplotlib >= 3.9 behavior changes that make pyplot.colorbar(...)
raise when it cannot infer which Axes to attach to.

Context:
- Some workflows call `plt.colorbar(ScalarMappable(), ...)` without `ax=...`.
- Newer Matplotlib versions raise:
  "Unable to determine Axes to steal space for Colorbar..."

This file is auto-imported by Python on startup (via the `site` module) when the
repository root is on sys.path (true for GitHub Actions + local runs executed
from repo root).

Goal:
- Keep the workflow-first pipeline stable without refactoring plot scripts.
- Only intervene on this specific error and fall back to `plt.gca()`.

Opt-out:
- Set ASTROGRAPHANOMALY_DISABLE_SITECUSTOMIZE=1 to disable this patch.
"""

from __future__ import annotations

import os


def _patch_matplotlib_colorbar() -> None:
    if os.environ.get("ASTROGRAPHANOMALY_DISABLE_SITECUSTOMIZE", "").strip() == "1":
        return

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    old_colorbar = getattr(plt, "colorbar", None)
    if old_colorbar is None:
        return

    def colorbar(*args, **kwargs):  # noqa: ANN001
        try:
            return old_colorbar(*args, **kwargs)
        except ValueError as e:
            msg = str(e)
            needle = "Unable to determine Axes to steal space for Colorbar"
            if needle not in msg:
                raise

            # If caller didn't pass an Axes, fall back to current Axes.
            # This mirrors Matplotlib's prior implicit behavior.
            if kwargs.get("ax") is None:
                try:
                    kwargs["ax"] = plt.gca()
                    return old_colorbar(*args, **kwargs)
                except Exception:
                    # If fallback also fails, re-raise the original error.
                    raise e
            raise

    plt.colorbar = colorbar  # type: ignore[attr-defined]


_patch_matplotlib_colorbar()
