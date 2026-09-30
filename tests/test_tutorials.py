"""The teaching notebooks must build, parse, and not contain dead cells.

A tutorial that no longer runs is worse than no tutorial: it teaches the
reader something false about the code it sits next to. These tests are fast
(they build without executing); `physprior tutorials --execute` is what
actually runs them, and CI does that on push.
"""

from __future__ import annotations

import ast
import json

import pytest

from physprior.config import get_settings
from physprior.reporting.tutorials import TUTORIALS, build

ORDER = [f"T{i}" for i in range(1, 15)]


def test_the_course_covers_every_physics_topic_in_order():
    """The arc is deliberate: gravity (exact law) -> relativity (truncated
    law) -> quantum (no data at all). Losing one breaks the progression."""
    names = list(TUTORIALS)
    assert [n.split("_")[0] for n in names] == ORDER
    joined = " ".join(names).lower()
    for topic in ("gravity", "relativity", "quantum"):
        assert topic in joined, f"the course has no {topic} notebook"


@pytest.mark.parametrize("name", list(TUTORIALS))
def test_every_tutorial_builds_and_its_code_parses(name):
    nb = TUTORIALS[name]()
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 3, f"{name} has only {len(code)} code cells"
    for i, cell in enumerate(code):
        try:
            ast.parse(cell.source)
        except SyntaxError as exc:  # pragma: no cover - the assert reports it
            pytest.fail(f"{name} code cell {i}: {exc}")


@pytest.mark.parametrize("name", list(TUTORIALS))
def test_every_tutorial_explains_itself(name):
    """A notebook of code cells is a script, not a tutorial."""
    nb = TUTORIALS[name]()
    markdown = [c for c in nb.cells if c.cell_type == "markdown"]
    words = sum(len(c.source.split()) for c in markdown)
    assert len(markdown) >= len([c for c in nb.cells if c.cell_type == "code"]) - 2, (
        f"{name} has {len(markdown)} prose cells"
    )
    assert words > 250, f"{name} carries only {words} words of explanation"


def test_the_index_lists_every_tutorial():
    index = (get_settings().notebooks_dir / "tutorials" / "README.md").read_text()
    for name in TUTORIALS:
        assert name in index, f"the tutorials index does not link {name}"


def test_building_writes_the_notebooks(tmp_settings):
    build(execute=False)
    out = tmp_settings.notebooks_dir / "tutorials"
    written = sorted(p.stem for p in out.glob("*.ipynb"))
    assert written == sorted(TUTORIALS)


@pytest.mark.parametrize("name", list(TUTORIALS))
def test_executed_tutorials_have_no_error_cells(name):
    """If a tutorial has been executed, it must have run clean."""
    path = get_settings().notebooks_dir / "tutorials" / f"{name}.ipynb"
    if not path.exists():
        pytest.skip("tutorials not built yet -- run `physprior tutorials`")
    nb = json.loads(path.read_text())
    executed = any(c.get("outputs") for c in nb["cells"] if c["cell_type"] == "code")
    if not executed:
        pytest.skip(f"{name} has not been executed")
    errors = [
        o
        for c in nb["cells"]
        for o in c.get("outputs", [])
        if o.get("output_type") == "error"
    ]
    assert not errors, f"{name}: {[e.get('ename') for e in errors]}"
