"""The controlled test of when a physics prior actually helps.

The claim under test is not "a physics prior helps when the law is
incomplete" -- that is false as stated. It is that the prior helps when the
missing piece is **distinguishable from the law**, and these tests pin both
halves of that.
"""

from __future__ import annotations

import numpy as np
import pytest

from physprior.benchmark.neglected import GM_TRUE, NeglectedSystem, run_one


def test_the_two_shapes_differ_in_what_the_law_can_imitate():
    """`power` is one order higher in 1/r and nearly degenerate with the law;
    `bump` is localised and cannot be imitated at any GM."""
    r = np.linspace(1, 6, 200)
    power = NeglectedSystem(eps=0.4, noise=0.0, shape="power")
    bump = NeglectedSystem(eps=0.4, noise=0.0, shape="bump")
    # a rescaled law fits the power-law miss far better than the bump
    law = power.law(r)
    for sys_, tol in ((power, 0.25), (bump, 0.90)):
        miss = sys_.neglected(r)
        scale = float(np.sum(law * miss) / np.sum(law * law))
        residual = np.linalg.norm(miss - scale * law) / np.linalg.norm(miss)
        assert (residual < tol) == (sys_.shape == "power"), (
            f"{sys_.shape}: residual after absorbing into the law is {residual:.2f}"
        )


def test_an_unknown_shape_is_refused():
    with pytest.raises(ValueError, match="unknown shape"):
        NeglectedSystem(eps=0.1, noise=0.0, shape="magic").neglected(np.ones(3))


def test_eps_zero_is_the_exact_law():
    sys_ = NeglectedSystem(eps=0.0, noise=0.0)
    r = np.linspace(1, 6, 50)
    assert np.allclose(sys_.truth(r), sys_.law(r))
    assert sys_.neglected_fraction == 0.0


@pytest.mark.slow
def test_with_an_exact_law_the_plain_fit_wins():
    """The prior must not be claimed to help where there is nothing to add."""
    res = {r.arm: r for r in run_one(eps=0.0, noise=0.02, epochs=1200)}
    assert res["physics"].nrmse_in < res["pinn"].nrmse_in
    assert res["physics"].nrmse_in < res["nn"].nrmse_in


@pytest.mark.slow
def test_with_a_distinguishable_missing_term_the_pinn_wins():
    res = {
        r.arm: r
        for r in run_one(eps=0.4, noise=0.02, shape="bump", w_phys=0.01, epochs=2000)
    }
    assert res["pinn"].nrmse_in < res["physics"].nrmse_in
    assert res["pinn"].nrmse_in < res["nn"].nrmse_in


@pytest.mark.slow
def test_the_correction_learns_the_term_it_was_never_shown():
    """1.0 would be as good as predicting zero, so this must be well under it."""
    res = {
        r.arm: r
        for r in run_one(eps=0.4, noise=0.02, shape="bump", w_phys=0.01, epochs=2000)
    }
    assert res["pinn"].correction_error < 0.5


@pytest.mark.slow
def test_a_degenerate_missing_term_corrupts_the_constant_instead():
    """The counter-example, and the actual conclusion of the study: when the
    law can imitate what is missing, the fit absorbs it into the constant."""
    bump = {
        r.arm: r
        for r in run_one(eps=0.4, noise=0.02, shape="bump", w_phys=0.01, epochs=2000)
    }
    power = {
        r.arm: r
        for r in run_one(eps=0.4, noise=0.02, shape="power", w_phys=0.01, epochs=2000)
    }
    assert power["pinn"].gm_error_pct > 5 * bump["pinn"].gm_error_pct
    assert power["pinn"].correction_error > bump["pinn"].correction_error


def test_the_arms_are_compared_at_comparable_capacity():
    """`pinn` carries one extra parameter -- the constant -- and otherwise the
    same network as `nn`."""
    res = {r.arm: r for r in run_one(eps=0.2, noise=0.02, epochs=50)}
    assert res["pinn"].n_params == res["nn"].n_params + 1
    assert res["physics"].n_params == 1
    assert GM_TRUE == 1.0
