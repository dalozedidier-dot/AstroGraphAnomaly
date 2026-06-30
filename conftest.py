"""Pytest bootstrap.

Make the package importable from a fresh checkout without requiring an
editable install (``pip install -e .``). Adding ``src/`` to ``sys.path`` keeps
``pytest`` runnable directly from the repo root (GitHub web + Colab friendly),
mirroring how ``run_workflow.py`` resolves imports.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
