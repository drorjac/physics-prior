"""The summary notebook: it builds, its code parses, and every plot draws from
the committed results/ files without network access or retraining."""

from __future__ import annotations

import ast

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from physprior.reporting import summary as S

PLOTS = sorted({fn.__name__ for t in S.TASKS.values() for fn in (*t.data, *t.results)})


def test_notebook_code_parses():
    nb = S.notebook()
    sources = [c.source for c in nb.cells if c.cell_type == "code"]
    assert sources
    for src in sources:
        ast.parse(src)


def test_notebook_has_every_task_and_the_chooser():
    nb = S.notebook()
    text = "\n".join(c.source for c in nb.cells)
    assert 'TASK = "lorenz"' in text and "S.show(TASK)" in text
    for fn in PLOTS:
        assert f"S.{fn}()" in text
    assert "## In progress" in text


def test_build_writes_to_notebooks_dir(tmp_path, monkeypatch):
    from physprior.config import reset_settings

    monkeypatch.setenv("PHYSPRIOR_NOTEBOOKS_DIR", str(tmp_path))
    reset_settings()
    try:
        S.build(execute=False)
    finally:
        monkeypatch.delenv("PHYSPRIOR_NOTEBOOKS_DIR")
        reset_settings()
    assert (tmp_path / "summary.ipynb").exists()


@pytest.mark.parametrize("name", PLOTS)
def test_plot_runs_on_committed_results(name):
    fig = getattr(S, name)()
    # the H7 plots return None, with a note, while their tables are missing
    if fig is not None:
        assert fig.axes
        plt.close(fig)


@pytest.mark.parametrize("task", list(S.TASKS))
def test_conclusion_is_one_line_of_computed_text(task):
    text = S.TASKS[task].conclusion()
    assert text and "\n" not in text


def test_show_rejects_unknown_task():
    with pytest.raises(KeyError):
        S.show("no-such-task")
