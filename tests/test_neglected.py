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


# --- the same question for a differential law ----------------------------
def test_the_two_ode_shapes_depend_on_different_variables():
    """Which variable the missing force depends on is the whole question:
    a harmonic model can shift omega to absorb an angle-dependent term, and
    can do nothing at all about a velocity-dependent one."""
    from physprior.benchmark.neglected import NeglectedODE

    anh = NeglectedODE(amplitude=1.0, noise=0.0, shape="anharmonic")
    damp = NeglectedODE(amplitude=1.0, noise=0.0, shape="damping")
    theta = np.linspace(-1, 1, 50)
    # the anharmonic force is a function of angle alone
    assert np.any(np.abs(anh.missing_force(theta)) > 0)
    # damping vanishes identically when the velocity is not supplied
    assert np.allclose(damp.missing_force(theta), 0.0)
    assert np.any(np.abs(damp.missing_force(theta, np.ones_like(theta))) > 0)


def test_an_unknown_ode_shape_is_refused():
    from physprior.benchmark.neglected import NeglectedODE

    with pytest.raises(ValueError, match="unknown shape"):
        NeglectedODE(amplitude=1.0, noise=0.0, shape="magic").missing_force(0.5)


def test_the_anharmonic_period_grows_with_amplitude():
    """The observable consequence of the small-angle step, in closed form."""
    from physprior.benchmark.neglected import NeglectedODE

    small = NeglectedODE(amplitude=np.radians(5), noise=0.0, shape="anharmonic")
    large = NeglectedODE(amplitude=np.radians(120), noise=0.0, shape="anharmonic")
    assert small.true_period == pytest.approx(2 * np.pi, rel=1e-3)
    assert large.true_period > 1.2 * small.true_period


@pytest.mark.slow
def test_a_conservative_model_cannot_decay_at_any_omega():
    """The `physics` arm's error on a damped pendulum is a property of the
    model, not of the fit -- so it barely moves with amplitude."""
    from physprior.benchmark.neglected import run_ode

    lo = {r.arm: r for r in run_ode(np.radians(30), shape="damping", epochs=1500)}
    hi = {r.arm: r for r in run_ode(np.radians(120), shape="damping", epochs=1500)}
    assert lo["physics"].nrmse_in == pytest.approx(hi["physics"].nrmse_in, rel=0.15)
    assert hi["pinn"].nrmse_in < 0.3 * hi["physics"].nrmse_in


# --- and once more for a partial differential law ------------------------
def test_a_diffusive_missing_term_biases_alpha_by_exactly_eps():
    """The sharpest identifiability statement in the package, in closed form.

    `eps * alpha * u_xx` is MORE OF THE SAME OPERATOR, so a single rescaling
    alpha -> alpha (1 + eps) reproduces the truth exactly. The prediction is
    then flawless and the constant is wrong by precisely eps -- which is the
    dangerous case, because nothing about the fit looks wrong.
    """
    from physprior.benchmark.neglected import NeglectedPDE, _pde_fit_physics

    baseline = _pde_fit_physics(NeglectedPDE(eps=0.0, noise=0.0, shape="diffusive"))
    for eps in (0.2, 0.4):
        sys_ = NeglectedPDE(eps=eps, noise=0.0, shape="diffusive")
        alpha_hat = _pde_fit_physics(sys_)
        # measured against the eps = 0 estimate, which carries the scheme's
        # own discretisation bias and would otherwise be charged to eps
        assert alpha_hat / baseline - 1.0 == pytest.approx(eps, rel=0.15)
        assert sys_.expected_alpha_bias == eps


def test_an_advective_missing_term_leaves_alpha_alone():
    """Drift is orthogonal to `u_xx` in the least-squares projection, so it
    does not bias the constant at all -- it makes the MODEL wrong instead.
    The two failure signatures are opposite, which is the point."""
    from physprior.benchmark.neglected import NeglectedPDE, _pde_fit_physics

    baseline = _pde_fit_physics(NeglectedPDE(eps=0.0, noise=0.0, shape="advective"))
    drifted = _pde_fit_physics(NeglectedPDE(eps=0.4, noise=0.0, shape="advective"))
    assert abs(drifted / baseline - 1.0) < 0.05
    assert np.isnan(
        NeglectedPDE(eps=0.4, noise=0.0, shape="advective").expected_alpha_bias
    )


def test_the_advective_residual_is_far_larger_than_the_diffusive_one():
    """What the modelled equation cannot reproduce, at its own best alpha."""
    from physprior.benchmark.neglected import NeglectedPDE, _pde_fit_physics

    peaks = {}
    for shape in ("diffusive", "advective"):
        sys_ = NeglectedPDE(eps=0.3, noise=0.0, shape=shape)
        best = type(sys_)(eps=0.0, noise=0.0, shape=shape, alpha=_pde_fit_physics(sys_))
        _, _, truth = sys_.solve(60)
        _, _, model = best.solve(60)
        peaks[shape] = float(np.max(np.abs(truth - model)))
    assert peaks["advective"] > 4 * peaks["diffusive"]


def test_an_unknown_pde_shape_is_refused():
    from physprior.benchmark.neglected import NeglectedPDE

    with pytest.raises(ValueError, match="unknown shape"):
        NeglectedPDE(eps=0.1, noise=0.0, shape="magic").solve(5)


def test_the_pde_solution_conserves_its_boundaries():
    from physprior.benchmark.neglected import NeglectedPDE

    _, _, u = NeglectedPDE(eps=0.2, noise=0.0).solve(20)
    assert np.allclose(u[:, 0], 0.0, atol=1e-9)
    assert np.allclose(u[:, -1], 0.0, atol=1e-9)


def test_the_pde_pinn_is_marked_as_not_converged():
    """It does not converge in this configuration, and a number from a fit
    that did not converge is not a measurement. The guard must fire."""
    from physprior.benchmark.neglected import (
        PDE_PINN_ALPHA_FLOOR,
        pde_pinn_converged,
    )

    # what it actually returns: alpha driven to the floor, field unfitted
    assert not pde_pinn_converged(0.001, 0.5)
    # a fit that worked would pass
    assert pde_pinn_converged(0.051, 0.01)
    # and the guard is not vacuous in either direction
    assert not pde_pinn_converged(0.051, 0.5)
    assert not pde_pinn_converged(0.001, 0.01)
    assert PDE_PINN_ALPHA_FLOOR < 1.0


# ---------------------------------------------------------------------------
# Fitting a field is not the same as differentiating it


def test_implied_alpha_is_exact_on_the_exact_field():
    """The estimator itself must be unbiased, or nothing below means anything.

    alpha = argmin |u_t - a u_xx|^2 recovers 0.05 to five decimals on the
    analytic solution. So when a fitted network returns 0.033, the estimator
    is not what is wrong -- the network's derivatives are.
    """
    torch = pytest.importorskip("torch")
    from physprior.benchmark.neglected import NeglectedPDE

    sys_ = NeglectedPDE(eps=0.0, noise=0.0, shape="diffusive")
    length, w = float(sys_.length), 0.08 * float(sys_.length)
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.uniform(0, length, 4000), dtype=torch.float64).requires_grad_(
        True
    )
    t = torch.tensor(
        rng.uniform(0, sys_.t_max, 4000), dtype=torch.float64
    ).requires_grad_(True)
    s2 = w**2 + 4.0 * sys_.alpha * t
    u = (w / torch.sqrt(s2)) * torch.exp(-((x - 0.5 * length) ** 2) / s2)
    u_t = torch.autograd.grad(u.sum(), t, create_graph=True)[0].detach()
    u_x = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0].detach()
    alpha = float((u_t * u_xx).sum() / (u_xx**2).sum())
    assert abs(alpha - sys_.alpha) < 1e-6, (
        f"the least-squares estimator is biased on the EXACT field: {alpha}"
    )


@pytest.mark.slow
def test_curvature_penalty_reduces_the_excess_curvature():
    """The mechanism, at a size a test can afford.

    Not asserting a recovered alpha -- that needs the full run. Asserting the
    link the fix rests on: the penalty lowers mean |u_xx| toward the exact
    field's value, which is what moves alpha.
    """
    pytest.importorskip("torch")
    from physprior.benchmark.neglected import derivative_accuracy_study

    out = derivative_accuracy_study(n=200, epochs=600, weights=(0.0, 3e-2))
    assert out["curvature"][1] < out["curvature"][0], (
        "the curvature penalty did not reduce mean |u_xx|: "
        f"{out['curvature'][0]:.3f} -> {out['curvature'][1]:.3f}"
    )
    assert out["uxx_net"].shape == out["uxx_exact"].shape


def test_derivative_accuracy_figure_is_finite():
    from physprior.viz import plots as P

    x = np.linspace(0, 1, 50)
    fig = P.fig_derivative_accuracy(
        x,
        np.exp(-((x - 0.5) ** 2)),
        np.exp(-((x - 0.5) ** 2)) + 1e-3,
        -np.cos(x),
        -np.cos(x) * 1.5,
        ([0.0, 0.01, 0.03], [0.033, 0.045, 0.055]),
        0.05,
        title="check",
    )
    assert len(fig.axes) == 3
    # the bug that shipped a 100,991 px figure: a data value used as a
    # coordinate. Width is in inches and must stay sane.
    assert fig.get_size_inches()[0] < 20


def test_derivative_study_reference_is_the_walled_solution():
    """The reference field in `derivative_accuracy_study` must satisfy the
    solver's zero walls. The free-space Gaussian misses by ~0.06 at t_max;
    the method-of-images form matches the solver to its own grid error."""
    torch = pytest.importorskip("torch")
    from physprior.benchmark.neglected import NeglectedPDE, pde_exact_diffusive

    sys_ = NeglectedPDE(eps=0.0, noise=0.0, shape="diffusive")
    t, x, u = sys_.solve(30)
    xx, tt = np.meshgrid(x, t)
    ex = pde_exact_diffusive(
        torch.tensor(xx), torch.tensor(tt), sys_.alpha, sys_.length, 0.08
    ).numpy()
    assert np.max(np.abs(ex - u)) < 2e-3
    assert np.allclose(ex[:, [0, -1]], 0.0, atol=1e-12)
