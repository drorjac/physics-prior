"""The Lorenz study: its numerics, its arms, and the claims its page makes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from physprior.config import get_settings  # noqa: E402
from physprior.lorenz import classical as C  # noqa: E402
from physprior.lorenz import metrics as M  # noqa: E402
from physprior.lorenz import pinn as LP  # noqa: E402
from physprior.lorenz import system as S  # noqa: E402

# ---------------------------------------------------------------------------
# numerics


@pytest.mark.parametrize("fourier", [0, 4])
@pytest.mark.parametrize("mode", ["jvp", "forward"])
def test_derivative_modes_agree_with_autograd(mode, fourier):
    torch.manual_seed(0)
    net = LP.LorenzNet(16, 3, fourier)
    s = torch.linspace(-1, 1, 33, dtype=torch.float64)[:, None]
    n_ref, d_ref = LP.derivative_fn(net, "autograd")(s)
    n, d = LP.derivative_fn(net, mode)(s)
    assert torch.allclose(n, n_ref, atol=1e-13)
    assert torch.allclose(d, d_ref, atol=1e-12)


def test_forward_mode_matches_finite_differences():
    torch.manual_seed(1)
    net = LP.LorenzNet(16, 2, 3)
    s = torch.linspace(-0.9, 0.9, 7, dtype=torch.float64)[:, None]
    h = 1e-6
    fd = (net(s + h) - net(s - h)) / (2 * h)
    _, d = net.forward_with_derivative(s)
    assert torch.allclose(d, fd, atol=1e-7)


def test_rk4_is_fourth_order_and_its_step_is_converged():
    rows = S.rk4_convergence(t_end=1.0)
    orders = [r["order"] for r in rows if "order" in r]
    assert min(orders) > 3.5
    used = next(r for r in rows if r["h"] == S.RK4_H)
    assert used["err"] < 1e-6


def test_observations_are_reproducible_and_noise_free_at_zero():
    a = S.observe(5, n_obs=12, noise=0.0, t_end=1.0, n_dense=101)
    b = S.observe(5, n_obs=12, noise=0.0, t_end=1.0, n_dense=101)
    np.testing.assert_array_equal(a.y_obs, b.y_obs)
    truth = S.reference(a.u0, a.t_obs)
    np.testing.assert_allclose(a.y_obs, truth, atol=1e-12)
    noisy = S.observe(5, n_obs=12, noise=0.1, t_end=1.0, n_dense=101)
    np.testing.assert_array_equal(noisy.t_obs, a.t_obs)
    assert not np.allclose(noisy.y_obs, a.y_obs)


def test_regression_recovers_the_constants_from_exact_derivatives():
    obs = S.observe(5, n_obs=10, noise=0.0, t_end=1.0, n_dense=201)
    th = C.regress_theta(obs.u_dense, obs.du_dense)
    np.testing.assert_allclose(th, S.THETA_TRUE, rtol=1e-10)


def test_growth_rate_of_a_pure_exponential():
    t = np.linspace(0, 10, 200)
    assert S.growth_rate(t, 1e-6 * np.exp(0.7 * t), 1e-5, 1e-1) == pytest.approx(0.7)


def test_valid_time_counts_to_the_first_crossing():
    t = np.linspace(3.0, 5.0, 201)
    err = np.where(t < 4.0, 0.1, 1.0)
    assert M.valid_time(err, t, 3.0) == pytest.approx(1.0 * S.LAMBDA1, rel=1e-2)


def test_theta_errors_are_relative():
    e = M.theta_errors(S.THETA_TRUE * np.array([1.1, 1.0, 0.8]))
    assert e["err_sigma"] == pytest.approx(0.1)
    assert e["err_beta"] == pytest.approx(0.2)
    assert e["err_theta"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# arms


def test_shooting_from_the_truth_stays_there_on_clean_data():
    obs = S.observe(5, n_obs=20, noise=0.0, t_end=1.0, n_dense=101)
    sf = C.single_shooting(obs, S.THETA_TRUE, obs.u0, max_nfev=20)
    assert M.theta_errors(sf.theta)["err_theta"] < 1e-4
    assert sf.cost < 1e-6


def test_multiple_shooting_on_clean_data():
    obs = S.observe(5, n_obs=30, noise=0.0, t_end=1.0, n_dense=101)
    sf = C.multiple_shooting(obs, n_seg=4, max_nfev=60)
    assert M.theta_errors(sf.theta)["err_theta"] < 1e-3


def test_a_short_fit_runs_and_records_every_loss_term():
    obs = S.observe(5, n_obs=20, noise=0.05, t_end=1.0, n_dense=101)
    cfg = LP.PinnConfig(
        steps=30,
        warmup=5,
        ramp=5,
        lbfgs=5,
        width=8,
        depth=2,
        fourier=2,
        n_col=32,
        record_every=5,
        n_lbfgs_grid=64,
    )
    f = LP.fit(obs, cfg, seed=0)
    assert f.theta is not None and f.theta.shape == (3,)
    for k in ("data", "phys", "phys_x", "phys_y", "phys_z", "w", *S.THETA_NAMES):
        assert len(f.history[k]) == len(f.history["step"])
    assert "lbfgs" in f.history["phase"]
    assert f.predict(obs.t_dense).shape == (101, 3)
    assert f.derivative(obs.t_dense).shape == (101, 3)


def test_the_black_box_has_no_constants_and_no_physics_term():
    obs = S.observe(5, n_obs=20, noise=0.05, t_end=1.0, n_dense=101)
    f = LP.fit(
        obs,
        LP.PinnConfig(physics=False, steps=10, width=8, depth=2, fourier=0, lbfgs=0),
        seed=0,
    )
    assert f.theta is None
    assert all(np.isnan(f.history["phys"]))


def test_the_ramp_holds_the_physics_weight_at_zero_during_warm_up():
    obs = S.observe(5, n_obs=20, noise=0.05, t_end=1.0, n_dense=101)
    cfg = LP.PinnConfig(
        steps=40,
        warmup=20,
        ramp=10,
        lbfgs=0,
        width=8,
        depth=2,
        fourier=0,
        n_col=16,
        record_every=1,
    )
    h = LP.fit(obs, cfg, seed=0).history
    w = np.asarray(h["w"])
    assert np.all(w[:20] == 0) and w[-1] == pytest.approx(cfg.w_phys)


def test_the_cli_knows_the_study():
    from physprior.cli import build_parser

    args = build_parser().parse_args(["lorenz", "--quick", "--no-gif"])
    assert args.command == "lorenz" and args.quick and args.no_gif


# ---------------------------------------------------------------------------
# the claims docs/lorenz/README.md makes, against the committed results


def _committed() -> bool:
    return (get_settings().results_dir / "lorenz" / "main.csv").exists()


needs_results = pytest.mark.skipif(not _committed(), reason="no results/lorenz yet")


@pytest.fixture(scope="module")
def findings():
    from physprior.lorenz.doc import findings as f

    return f()


@needs_results
def test_claim_pinn_beats_the_black_box_on_every_seed(findings):
    assert findings["wins_state"] == findings["n_seeds"]
    assert findings["state_ratio"] > 5  # "about an order of magnitude"
    assert findings["deriv_ratio"] > 5


@needs_results
def test_claim_single_shooting_fails_where_multiple_shooting_and_pinn_do_not(findings):
    assert findings["theta_shooting"] > 10 * findings["theta_pinn"]
    assert findings["theta_ms"] < 0.05 and findings["theta_pinn"] < 0.05


@needs_results
def test_claim_the_vanilla_pinn_fails_and_the_recipe_fixes_it(findings):
    worst = findings["ladder_worst"]
    assert findings["vanilla_fails"] >= 2
    assert worst["vanilla"] > 0.1
    assert worst["+ Fourier features"] < worst["+ warm-up and ramp"]
    assert findings["ladder_gain"] > 5
    # it stalls without fitting the data: far above the noise floor
    assert findings["vanilla_data_over_floor"] > 10


@needs_results
def test_claim_the_warm_up_fits_the_noise_then_the_constants_snap(findings):
    assert findings["warmup_data_min_over_floor"] < 1
    assert 0 < findings["steps_to_5pct"] < 1000


@needs_results
def test_claim_the_noise_factor_is_largest_on_clean_data(findings):
    factors = [findings["noise_ratio"][k] for k in sorted(findings["noise_ratio"])]
    assert factors[0] == max(factors)
    assert all(a > b for a, b in itertools.pairwise(factors))


@needs_results
def test_claim_with_ten_points_only_the_pinn_recovers_the_constants(findings):
    n = min(findings["budget_theta_pinn"])
    assert findings["budget_theta_pinn"][n] < 0.05
    assert findings["budget_theta_ms"][n] > 5 * findings["budget_theta_pinn"][n]
    assert findings["budget_theta_nn_regress"][n] > 5 * findings["budget_theta_pinn"][n]


@needs_results
def test_claim_multiple_shooting_trajectory_worst_seed(findings):
    assert findings["state_ms_worst"] > findings["state_pinn_worst"]
    assert findings["shooting_fails"] >= 2


@needs_results
def test_claim_gradnorm_helps_vanilla_and_causal_adds_nothing(findings):
    assert findings["ladder_gradnorm"] < findings["ladder_vanilla"] / 2
    assert findings["ladder_causal"] == pytest.approx(findings["ladder_final"], rel=0.1)


@needs_results
def test_claim_forward_mode_is_the_fastest_derivative(findings):
    assert findings["speed_forward"] < findings["speed_autograd"]
    assert findings["speed_forward"] < findings["speed_jvp"]
    assert findings["compile_gain"] > 1


@needs_results
def test_claim_the_measured_lyapunov_exponent(findings):
    assert findings["lambda_fit"] == pytest.approx(findings["lambda_ref"], rel=0.1)


@needs_results
def test_claim_the_page_is_up_to_date(tmp_path):
    from physprior.lorenz.doc import DOC, render_doc

    fresh = render_doc(tmp_path / "page.md")
    committed = (get_settings().root / "docs" / DOC).read_text()
    assert fresh == committed, (
        "docs/lorenz/README.md is stale: run physprior lorenz --doc-only"
    )
