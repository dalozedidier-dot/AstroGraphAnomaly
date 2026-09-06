#!/usr/bin/env python3
# Concatenate the implementation parts so the theme CSS can live in pages_theme.css.
from __future__ import annotations

from pathlib import Path

_here = Path(__file__).resolve().parent
_code = (_here / "pages_impl_a.py").read_text(encoding="utf-8") + (_here / "pages_impl_b.py").read_text(encoding="utf-8")
exec(compile(_code, str(Path(__file__)), "exec"))
