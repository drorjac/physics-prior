"""T11 builds and every code cell parses (execution is `slow`)."""

from __future__ import annotations

import ast

import pytest

from physprior.reporting.tutorials_sr import t11_how_symbolic_regression_works


def test_t11_builds_and_its_code_parses():
    nb = t11_how_symbolic_regression_works()
    code = [c for c in nb.cells if c.cell_type == "code"]
    assert len(code) >= 8
    for cell in code:
        ast.parse(cell.source)
    assert nb.metadata["title"].startswith("T11")


@pytest.mark.slow
def test_t11_executes(tmp_path):
    nbclient = pytest.importorskip("nbclient")
    from physprior.config import get_settings

    nb = t11_how_symbolic_regression_works()
    nbclient.NotebookClient(
        nb,
        timeout=900,
        kernel_name="physprior",
        resources={"metadata": {"path": str(get_settings().root)}},
    ).execute()
