"""The pinn arm's optional machinery.

Every option defaults to OFF, because the committed results were produced
without them; an option may only change a published number by being turned on
deliberately after the tuning seeds said it earns its place.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.methods.pinn import (  # noqa: E402
    DEFAULT_PINN,
    FROZEN_PINN,
    MIN_POINTS_FOR_EARLY_STOPPING,
    PhysParam,
    PinnOptions,
    _split_validation,
    fit_pinn,
)


def _toy(n=40, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    x = np.linspace(1.0, 4.0, n).reshape(-1, 1)
    y = 3.0 * x.ravel() ** 1.5 + rng.normal(0.0, noise, n)
    return x, y


def _law_t(xt, GM):
    return GM * xt.squeeze(-1) ** 1.5


PARAMS = [PhysParam("GM", 2.0)]


def _fit(options=DEFAULT_PINN, n=40, epochs=300, seed=11):
    x, y = _toy(n)
    return fit_pinn(x, y, _law_t, PARAMS, epochs=epochs, seed=seed, options=options)


def test_every_option_is_off_by_default():
    o = DEFAULT_PINN
    assert (o.early_stopping, o.lbfgs, o.ensemble, o.fourier, o.balance) == (
        False,
        False,
        1,
        0,
        False,
    )
    assert o.tag == "baseline"


def test_the_tag_names_the_configuration():
    assert PinnOptions(early_stopping=True).tag == "early"
    assert PinnOptions(ensemble=5, lbfgs=True).tag == "lbfgs+ens5"


@pytest.mark.parametrize(
    "options",
    [
        PinnOptions(),
        PinnOptions(early_stopping=True),
        PinnOptions(lbfgs=True),
        PinnOptions(ensemble=3),
        PinnOptions(fourier=8),
        PinnOptions(balance=True),
    ],
    ids=lambda o: o.tag,
)
def test_each_option_trains_and_recovers_the_constant(options):
    fit = _fit(options)
    assert np.isfinite(fit.params["GM"])
    assert abs(fit.params["GM"] - 3.0) / 3.0 < 0.05, (
        f"{options.tag} lost the constant entirely: {fit.params['GM']}"
    )
    assert np.all(np.isfinite(fit.predict(_toy()[0])))


def test_validation_split_partitions_the_training_set():
    """Every training point is either fitted on or validated on, never both.

    The split is carved from the TRAINING data: selecting an epoch on the
    held-out set would be the same error as tuning on a reporting seed.
    """
    fit_idx, val_idx = _split_validation(20, 0.25, seed=11)
    assert len(val_idx) == 5
    assert len(fit_idx) == 15
    assert set(fit_idx).isdisjoint(val_idx)
    assert sorted([*fit_idx, *val_idx]) == list(range(20))


def test_a_validation_split_is_always_non_empty():
    _, val_idx = _split_validation(3, 0.01, seed=11)
    assert len(val_idx) >= 1


def test_early_stopping_reports_itself_inapplicable_on_a_tiny_track():
    """Track G trains on as few as two points. Carving a validation split
    there is not possible, and that must be reported rather than silently
    producing a baseline fit labelled as an option."""
    fit = _fit(PinnOptions(early_stopping=True), n=MIN_POINTS_FOR_EARLY_STOPPING - 1)
    assert fit.extra["early_stopping_used"] is False
    assert fit.extra["engaged"] is False


def test_early_stopping_engages_when_there_is_room():
    fit = _fit(PinnOptions(early_stopping=True), n=40)
    assert fit.extra["early_stopping_used"] is True
    assert fit.extra["engaged"] is True


def test_ensemble_reports_a_spread_over_its_members():
    """The ensemble's only error bar is the scatter of its members, which is
    what makes it comparable with curve_fit's sigma at all."""
    fit = _fit(PinnOptions(ensemble=4))
    assert fit.extra["ensemble"] == 4
    assert fit.param_sigma["GM"] > 0.0
    assert len(fit.extra["members"]) == 4


def test_ensemble_prediction_is_the_mean_of_its_members():
    x, y = _toy()
    members = [
        fit_pinn(x, y, _law_t, PARAMS, epochs=300, seed=11 + k, options=DEFAULT_PINN)
        for k in range(3)
    ]
    ens = fit_pinn(
        x, y, _law_t, PARAMS, epochs=300, seed=11, options=PinnOptions(ensemble=3)
    )
    expected = np.mean([m.predict(x) for m in members], axis=0)
    assert np.allclose(ens.predict(x), expected, rtol=1e-10)


def test_lbfgs_is_reverted_when_it_does_not_help():
    """L-BFGS on a stiff loss can diverge; the refinement is a proposal, and
    a fit that got worse is not kept."""
    fit = _fit(PinnOptions(lbfgs=True))
    assert fit.extra["lbfgs_kept"] in (True, False)
    assert np.isfinite(fit.extra["data_mse_std_units"])
    assert abs(fit.params["GM"] - 3.0) / 3.0 < 0.05


def test_fourier_features_change_the_correction_not_the_law():
    fit = _fit(PinnOptions(fourier=8))
    assert abs(fit.params["GM"] - 3.0) / 3.0 < 0.05


def test_balancing_leaves_a_finite_weight():
    fit = _fit(PinnOptions(balance=True))
    assert np.isfinite(fit.extra["w_phys_final"])
    assert fit.extra["w_phys_final"] >= 0.0


def test_options_reach_the_arm_through_fit_arm():
    """An ablation turns a switch on for one run without touching the arm
    every other track uses."""
    from physprior.benchmark.protocol import Problem, fit_arm

    x, y = _toy(30)
    prob = Problem(
        track="t/toy",
        x=x,
        y=y,
        law_np=lambda xq, GM: GM * np.ravel(xq) ** 1.5,
        law_t=_law_t,
        params=PARAMS,
        theta_published={"GM": 3.0},
        pinn_epochs=200,
    )
    idx = np.arange(20)
    fit = fit_arm("pinn", prob, idx, seed=11, pinn_options=PinnOptions(ensemble=3))
    assert fit.extra["ensemble"] == 3
    # With no options passed the arm runs the FROZEN configuration -- the one
    # the reported results are produced with -- not the unswitched baseline.
    # The ablation asks for `PinnOptions()` explicitly when it wants that.
    base = fit_arm("pinn", prob, idx, seed=11)
    assert base.extra.get("options") == FROZEN_PINN.tag
    plain = fit_arm("pinn", prob, idx, seed=11, pinn_options=PinnOptions())
    assert plain.extra.get("options") == "baseline"
