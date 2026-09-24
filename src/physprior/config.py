"""Runtime configuration: where things are read from and written to.

Paths are resolved once, here, and every one can be overridden by an
environment variable so the package works the same from a checkout, from an
installed wheel, or in CI with a scratch directory:

    PHYSPRIOR_DATA_DIR      raw downloads          (default <root>/data)
    PHYSPRIOR_RESULTS_DIR   committed results      (default <root>/results)
    PHYSPRIOR_FIGURES_DIR   generated figures      (default <root>/figures)
    PHYSPRIOR_CACHE_DIR     symbolic-search cache  (default <root>/.cache)
    PHYSPRIOR_NOTEBOOKS_DIR notebooks              (default <root>/notebooks)
    PHYSPRIOR_OFFLINE       "1" to forbid network access entirely

`<root>` is the repository root when running from a checkout, and the current
working directory otherwise -- an installed copy must not try to write inside
site-packages.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


def _repo_root() -> Path:
    """The checkout root, or the working directory for an installed copy."""
    # <root>/src/physprior/config.py -> <root>
    candidate = _PACKAGE_DIR.parent.parent
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return Path.cwd()


def _path_from_env(var: str, default: Path) -> Path:
    raw = os.environ.get(var)
    return Path(raw).expanduser().resolve() if raw else default


@dataclass(frozen=True)
class Settings:
    """Resolved paths and switches. Immutable; build a new one to change it."""

    root: Path
    data_dir: Path
    results_dir: Path
    figures_dir: Path
    cache_dir: Path
    notebooks_dir: Path
    offline: bool

    @property
    def raw_dir(self) -> Path:
        """Downloads, exactly as served, never edited."""
        return self.data_dir / "raw"

    @property
    def sr_cache_dir(self) -> Path:
        """Symbolic-regression search cache, keyed by data + configuration."""
        return self.cache_dir / "sr"

    def results(self, problem: str) -> Path:
        path = self.results_dir / problem
        path.mkdir(parents=True, exist_ok=True)
        return path

    def figures(self, problem: str) -> Path:
        path = self.figures_dir / problem
        path.mkdir(parents=True, exist_ok=True)
        return path

    def ensure_dirs(self) -> None:
        for path in (
            self.raw_dir,
            self.results_dir,
            self.figures_dir,
            self.sr_cache_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The process-wide settings. Cached; call `reset_settings()` in tests."""
    root = _repo_root()
    return Settings(
        root=root,
        data_dir=_path_from_env("PHYSPRIOR_DATA_DIR", root / "data"),
        results_dir=_path_from_env("PHYSPRIOR_RESULTS_DIR", root / "results"),
        figures_dir=_path_from_env("PHYSPRIOR_FIGURES_DIR", root / "figures"),
        cache_dir=_path_from_env("PHYSPRIOR_CACHE_DIR", root / ".cache"),
        notebooks_dir=_path_from_env("PHYSPRIOR_NOTEBOOKS_DIR", root / "notebooks"),
        offline=os.environ.get("PHYSPRIOR_OFFLINE", "") == "1",
    )


def reset_settings() -> None:
    """Forget the cached settings, so a test can change the environment."""
    get_settings.cache_clear()


def short_path(path) -> str:
    """A path to print: relative to the project root when it is under it.

    `Path.relative_to` RAISES when it is not, and every output directory here
    is env-overridable precisely so it can be pointed somewhere else -- so
    the obvious `p.relative_to(root)` turns a redirected output directory
    into a crash after the work is already done. Caught by running
    `physprior neglected --quick` with PHYSPRIOR_RESULTS_DIR set to a
    scratch directory, which is the supported way to smoke-test a generator
    without overwriting committed artefacts.
    """
    from pathlib import Path

    p = Path(path)
    try:
        return str(p.relative_to(get_settings().root))
    except ValueError:
        return str(p)
