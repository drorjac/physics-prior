"""The device choice must never silently change the arithmetic.

There is one real trap here: MPS has no float64 kernels, so selecting it
demotes the package to single precision whether or not that was intended --
and the physics terms take SECOND derivatives, which is exactly where single
precision hurts. `resolve()` returns the dtype alongside the device so that
demotion cannot happen unannounced.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from physprior.methods import device  # noqa: E402


def test_available_reports_this_machine():
    have = device.available()
    assert have["cpu"] is True
    assert set(have) == {"cuda", "mps", "cpu", "threads"}
    assert have["threads"] >= 1


def test_default_is_float64():
    """Every number in results/ was produced in double precision."""
    _, dtype = device.resolve("cpu")
    assert dtype is torch.float64


def test_unavailable_device_warns_and_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(UserWarning, match="cuda requested"):
        dev, dtype = device.resolve("cuda")
    assert dev.type == "cpu"
    assert dtype is torch.float64


def test_mps_forces_float32_and_says_so(monkeypatch):
    """Not a preference -- MPS cannot do float64 at all."""
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setenv("PHYSPRIOR_DTYPE", "float64")
    with pytest.warns(UserWarning, match="SECOND derivatives"):
        dev, dtype = device.resolve("mps")
    assert dev.type == "mps"
    assert dtype is torch.float32


def test_dtype_can_be_traded_deliberately(monkeypatch):
    monkeypatch.setenv("PHYSPRIOR_DTYPE", "float32")
    _, dtype = device.resolve("cpu")
    assert dtype is torch.float32


def test_env_selects_the_device(monkeypatch):
    monkeypatch.setenv("PHYSPRIOR_DEVICE", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    dev, _ = device.resolve()
    assert dev.type == "cpu", "PHYSPRIOR_DEVICE must beat autodetection"


def test_describe_is_a_single_line():
    line = device.describe()
    assert "\n" not in line
    assert "device=" in line and "dtype=" in line


def test_info_reports_the_device(capsys):
    from physprior.cli import main

    assert main(["info"]) == 0
    assert "torch" in capsys.readouterr().out
