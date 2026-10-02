"""The data layer's contract, without touching the network."""

from __future__ import annotations

import pytest

from physprior.data.registry import DATASETS, loader
from physprior.exceptions import DataError, UnitError
from physprior.units import require, require_range


def test_every_registered_dataset_resolves():
    for name in DATASETS:
        assert callable(loader(name))


def test_unknown_dataset_names_the_alternatives():
    with pytest.raises(DataError) as exc:
        loader("nist/unobtainium")
    assert "nist/hydrogen" in str(exc.value)


def test_require_raises_uniterror_not_assertionerror():
    """`assert` vanishes under python -O; these guards must not."""
    require(True, "fine")
    with pytest.raises(UnitError, match="boom"):
        require(False, "boom")


def test_require_range_reports_the_value_and_the_bounds():
    require_range(1.5, 1.0, 2.0, "x")
    with pytest.raises(UnitError) as exc:
        require_range(5.0, 1.0, 2.0, "wavenumber", "cm^-1")
    message = str(exc.value)
    assert "5" in message and "cm^-1" in message and "[1, 2]" in message


def test_extraction_guards_raise_uniterror_so_handlers_stay_correct():
    """A too-strict SNR cut must raise `UnitError`, not `AssertionError`.

    The guards were `assert` statements once. When they became `UnitError`,
    two `except AssertionError` handlers in the threshold sweep stopped
    catching them, and a threshold that simply admits too few cycles aborted
    the whole study instead of being recorded as unusable.
    """
    import inspect

    from physprior.data.sources import gwosc
    from physprior.problems.relativity import discovery

    source = inspect.getsource(gwosc.extract_track)
    assert "require(" in source, "extract_track no longer guards its cycle count"

    handlers = inspect.getsource(discovery.threshold_calibration)
    assert "except AssertionError" not in handlers, (
        "threshold_calibration catches AssertionError, which the guards no longer raise"
    )
    assert "except UnitError" in handlers


class _Resp:
    def __init__(self, status: int, content: bytes = b""):
        self.status_code, self.content = status, content

    def raise_for_status(self):
        import requests

        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} Client Error", response=self)


def _online(monkeypatch, tmp_settings, resp):
    from physprior.config import reset_settings
    from physprior.data import cache

    monkeypatch.setenv("PHYSPRIOR_OFFLINE", "0")
    reset_settings()
    monkeypatch.setattr(cache.requests, "get", lambda *a, **k: resp)
    return cache


def test_download_error_carries_the_http_status(monkeypatch, tmp_settings):
    from physprior.exceptions import DownloadError

    cache = _online(monkeypatch, tmp_settings, _Resp(404))
    with pytest.raises(DownloadError) as exc:
        cache.cached_get("https://example.invalid/x", "x.bin")
    assert exc.value.status == 404


def test_download_is_written_whole_or_not_at_all(monkeypatch, tmp_settings):
    from physprior.config import get_settings

    cache = _online(monkeypatch, tmp_settings, _Resp(200, b"payload"))
    path = cache.cached_get("https://example.invalid/y", "y.bin")
    assert path.read_bytes() == b"payload"
    # no temporary file is left next to it
    assert [p.name for p in get_settings().raw_dir.iterdir()] == ["y.bin"]


def test_isd_treats_only_a_404_as_a_missing_file(monkeypatch, tmp_settings):
    from physprior.data.sources import isd
    from physprior.exceptions import DownloadError

    def gone(*a, **k):
        raise DownloadError("could not fetch: not found", status=404)

    def down(*a, **k):
        # an outage whose message happens to contain "404"
        raise DownloadError("could not fetch https://host:4040/404", status=503)

    tmp_settings.raw_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(isd, "cached_get", gone)
    assert isd.load_station_year("000000-00000", 2020) is None
    monkeypatch.setattr(isd, "cached_get", down)
    with pytest.raises(DownloadError):
        isd.load_station_year("111111-11111", 2020)
