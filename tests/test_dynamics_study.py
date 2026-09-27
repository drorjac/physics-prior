"""The dynamics study end to end, on a tiny budget."""

from __future__ import annotations

import json

import pytest

from physprior.dynamics import study


def _tiny():
    return study.StudyOptions(
        steps=5,
        batch=16,
        pde_batch=4,
        horizon=2,
        n_traj=2,
        n_test=2,
        n_val=1,
        seeds=(11,),
        tune=False,
        data_sizes=(1,),
        objective_ode=("pendulum",),
        objective_pde=("heat",),
        ode_systems=("pendulum",),
        pde_systems=("heat",),
        horizon_cap=12,
        workers=1,
    )


def test_tiny_study_writes_results_and_verdicts(tmp_settings, monkeypatch):
    monkeypatch.setattr(study, "reference_checks", lambda quick: {"ode": []})
    out = study.run(opts=_tiny())
    res = tmp_settings.results_dir / "dynamics"
    for name in (
        "ode_main.csv",
        "pde_main.csv",
        "objective.csv",
        "expectations.json",
        "meta.json",
        "curves.csv",
        "examples.npz",
    ):
        assert (res / name).exists(), name
    ver = json.loads((res / "expectations.json").read_text())
    assert set(ver) == {f"E{i}" for i in range(1, 10)}
    assert all(
        v["verdict"] in ("supported", "refuted", "not evaluated") for v in ver.values()
    )
    assert out["figures"]


def test_render_doc_after_tiny_study(tmp_settings, tmp_path, monkeypatch):
    from physprior.dynamics.doc import render_doc

    monkeypatch.setattr(study, "reference_checks", lambda quick: {"ode": []})
    study.run(opts=_tiny())
    text = render_doc(tmp_path / "README.md")
    assert "## Verdicts" in text
    assert "figures/dynamics/ode_error_vs_horizon.png" in text


def test_tutorial_builds_and_parses():
    import ast

    from physprior.reporting.tutorials_dynamics import t13_learning_the_update

    nb = t13_learning_the_update()
    for cell in nb.cells:
        if cell.cell_type == "code":
            ast.parse(cell.source)
    assert nb.metadata["title"].startswith("T13")


@pytest.mark.slow
def test_quick_mode_runs(tmp_settings):
    out = study.run(quick=True)
    assert out["meta"]["quick"] is True
