"""Release hygiene: the things that are embarrassing to get wrong in public.

Version numbers drift apart between `pyproject.toml`, `__init__.py` and
`CITATION.cff`; a rename leaves the old name in generated output; a console
script stops resolving. None of that is caught by testing the physics.
"""

from __future__ import annotations

import re
import tomllib

import pytest

import physprior
from physprior.config import get_settings

ROOT = get_settings().root

#: Names this project used before. A rename must not leave them behind, least
#: of all in generated output that gets committed.
SUPERSEDED_NAMES = ("project_0", "project0")

TEXT_SUFFIXES = {".py", ".md", ".toml", ".yml", ".yaml", ".cff", ".ipynb", ".sh"}
SKIP_DIRS = {
    ".venv",
    ".git",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "data",
    "__pycache__",
    "build",
    "dist",
}


def _tracked_text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if SKIP_DIRS & set(path.relative_to(ROOT).parts):
            continue
        yield path


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_version_is_consistent_everywhere(pyproject):
    version = physprior.__version__
    assert pyproject["project"]["version"] == version
    citation = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", citation, re.M), (
        f"CITATION.cff does not declare version {version}"
    )
    assert f"[{version}]" in (ROOT / "CHANGELOG.md").read_text(), (
        f"CHANGELOG.md has no section for {version}"
    )


def test_no_superseded_project_name_survives():
    """A rename is not finished until the old name is gone from the output."""
    offenders = []
    for path in _tracked_text_files():
        if path.name == "test_packaging.py":  # this file names them on purpose
            continue
        text = path.read_text(errors="replace")
        for name in SUPERSEDED_NAMES:
            if name in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {name!r}")
    assert not offenders, "\n".join(offenders)


def test_console_script_is_declared(pyproject):
    scripts = pyproject["project"].get("scripts", {})
    assert scripts.get("physprior") == "physprior.cli:main"


def test_optional_extras_cover_the_heavy_dependencies(pyproject):
    """torch and pysr must stay optional: the data layer and the classical
    fits are useful without either, and CI installs only what it needs."""
    extras = pyproject["project"]["optional-dependencies"]
    required = " ".join(pyproject["project"]["dependencies"])
    assert "torch" not in required and "pysr" not in required
    assert any("torch" in d for d in extras["nn"])
    assert any("pysr" in d for d in extras["sr"])


def test_package_is_typed(pyproject):
    assert (ROOT / "src" / "physprior" / "py.typed").is_file()
    assert "physprior" in pyproject["tool"]["setuptools"]["package-data"]


def test_readme_results_section_is_generated_not_typed():
    readme = (ROOT / "README.md").read_text()
    assert "<!-- RESULTS:START -->" in readme and "<!-- RESULTS:END -->" in readme
    start = readme.index("<!-- RESULTS:START -->")
    end = readme.index("<!-- RESULTS:END -->")
    assert "|" in readme[start:end], (
        "the README's generated results section is empty -- run `physprior report`"
    )
