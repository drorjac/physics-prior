"""The study plumbing: the law check, and a quick end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.symbolic.studies import LAWS, form_check, make_data, pysr_available


def test_form_check_accepts_the_law_in_any_spelling():
    law = LAWS["kepler"]
    for e in ("x0**1.5", "x0*sqrt(x0)", "1.02*sqrt(x0)**3"):
        assert form_check(e, law)["form_ok"], e
    # the right form, but a 2 % amplitude error: extrapolation still within 5 %
    assert form_check("1.02*sqrt(x0)**3", law)["recovered"]


def test_form_check_rejects_a_curve_that_only_fits_the_data():
    law = LAWS["planck"]
    # a good in-band fit without exp is not the law
    out = form_check("x0**2*(2.0 - 0.5*x0)/(1.0 + 0.1*x0**3)", law)
    assert not out["form_ok"] and not out["recovered"]
    assert form_check("x0**3/(exp(x0) - 1.0)", law)["recovered"]


def test_a_nested_form_with_bad_constants_is_not_recovered():
    law = LAWS["inverse_square"]
    out = form_check("5.93/x0**2 + 0.3*x0", law)
    assert out["form_ok"]  # 0.3 can be refitted to zero
    assert not out["recovered"]  # but as fitted it is wrong off the data


def test_make_data_noise_is_relative():
    _x, y, w, y0 = make_data(LAWS["bohr"], 12, 0.01, 11)
    assert np.std(y / y0 - 1) == pytest.approx(0.01, rel=0.6)
    np.testing.assert_allclose(w, 1 / y**2)


@pytest.mark.slow
def test_quick_study_runs_end_to_end(tmp_settings):
    from physprior.symbolic.figures import make_figures
    from physprior.symbolic.report import render_doc
    from physprior.symbolic.studies import run

    run(quick=True, parts=["growth", "tune", "noise", "vocabulary", "pareto", "sindy"])
    assert (tmp_settings.results_dir / "symbolic" / "recovery_noise.csv").exists()
    assert make_figures()
    paths = render_doc(docs_dir=tmp_settings.results_dir / "docs")
    assert all(p.exists() for p in paths)


@pytest.mark.sr
@pytest.mark.slow
def test_pysr_runner_returns_a_rescored_front(tmp_settings):
    if not pysr_available():
        pytest.skip("Julia off")
    from physprior.symbolic.studies import DEFAULT_OPS, run_pysr

    x, y, w, _ = make_data(LAWS["inverse_square"], 16, 0.0, 11)
    d = run_pysr(x, y, w, DEFAULT_OPS, 11, cfg={"niterations": 5})
    assert d["front"] and all("loss" in r for r in d["front"])
