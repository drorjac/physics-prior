"""Configuration: paths resolve, env vars win, settings are cacheable."""

from __future__ import annotations

from pathlib import Path

from physprior.config import get_settings, reset_settings


def test_defaults_live_under_the_repo_root():
    s = get_settings()
    assert (s.root / "pyproject.toml").is_file()
    assert s.raw_dir == s.data_dir / "raw"
    assert s.sr_cache_dir == s.cache_dir / "sr"


def test_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("PHYSPRIOR_RESULTS_DIR", str(tmp_path / "elsewhere"))
    reset_settings()
    try:
        assert get_settings().results_dir == (tmp_path / "elsewhere").resolve()
    finally:
        reset_settings()


def test_settings_are_cached():
    reset_settings()
    assert get_settings() is get_settings()


def test_results_and_figures_create_their_directories(tmp_settings):
    out = tmp_settings.results("gravity/kepler")
    assert out.is_dir()
    assert tmp_settings.figures("quantum/cmb").is_dir()
    assert isinstance(out, Path)
