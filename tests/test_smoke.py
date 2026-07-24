"""Smoke test confirming the package installs and imports correctly."""

import compass


def test_import():
    assert compass is not None
