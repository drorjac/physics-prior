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
