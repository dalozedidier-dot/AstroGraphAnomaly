from __future__ import annotations

import astrographanomaly
from astrographanomaly.cli import build_parser


def test_version() -> None:
    assert astrographanomaly.__version__ == "0.2.0"


def test_cli_has_csv_and_gaia_modes() -> None:
    opts = {a.dest for a in build_parser()._actions}
    assert "mode" in opts
