"""Shared fixtures.

The package is installed (`pip install -e .`), so there is no sys.path
surgery here. Tests that need the network or the Julia runtime are marked, so
CI can run the rest hermetically.
"""

from __future__ import annotations

import pytest

from physprior.config import get_settings, reset_settings


@pytest.fixture(autouse=True)
def _offline_guard(request):
    """Fail fast and clearly when an offline run reaches for the network."""
    settings = get_settings()
    if settings.offline and request.node.get_closest_marker("network"):
        pytest.skip("PHYSPRIOR_OFFLINE=1 and this test needs the network")


@pytest.fixture
def tmp_settings(tmp_path, monkeypatch):
    """Point every writable path at a temporary directory."""
    for var, sub in (
        ("PHYSPRIOR_RESULTS_DIR", "results"),
        ("PHYSPRIOR_FIGURES_DIR", "figures"),
        ("PHYSPRIOR_CACHE_DIR", "cache"),
        ("PHYSPRIOR_DATA_DIR", "data"),
    ):
        monkeypatch.setenv(var, str(tmp_path / sub))
    reset_settings()
    yield get_settings()
    reset_settings()
