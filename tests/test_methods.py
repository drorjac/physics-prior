"""Methods: each arm must recover a known answer on data where we know it."""

import numpy as np
import pytest

from physprior.benchmark.protocol import Problem, split_extrapolate, split_random
from physprior.methods.neural import train_mlp
from physprior.methods.pinn import PhysParam, fit_physics, fit_pinn, oracle
from physprior.methods.symbolic import power_law_exponent


def _toy(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(1, 4, n))
    return x, 2.5 * x**1.5 + rng.normal(0, 0.02, n)


def law_np(x, a):
    from physprior.util import first_column

    return a * first_column(x) ** 1.5


def law_t(x, a):
    return a * (x[:, 0] if x.ndim > 1 else x) ** 1.5


def test_physics_recovers_the_constant():
    x, y = _toy()
    f = fit_physics(x, y, law_np, [PhysParam("a", 1.0, lo=0, hi=10)])
    assert abs(f.params["a"] - 2.5) < 0.01
    assert f.param_sigma["a"] > 0


def test_pinn_recovers_it_too_and_the_correction_is_small():
    x, y = _toy()
    f = fit_pinn(x, y, law_t, [PhysParam("a", 1.0)], w_phys=10.0, epochs=1200)
    assert abs(f.params["a"] - 2.5) < 0.05
    assert f.extra["correction_rms_frac"] < 0.05


def test_physics_weight_dial_does_something():
    """w_phys = 0 must let the network absorb signal; large w_phys must not."""
    x, y = _toy()
    lo = fit_pinn(x, y, law_t, [PhysParam("a", 1.0)], w_phys=0.0, epochs=1200)
    hi = fit_pinn(x, y, law_t, [PhysParam("a", 1.0)], w_phys=1e3, epochs=1200)
    assert lo.extra["correction_rms_frac"] > hi.extra["correction_rms_frac"]


def test_oracle_has_no_free_parameters():
    f = oracle(law_np, {"a": 2.5})
    assert f.n_free == 0
    x, _ = _toy()
    assert np.allclose(f.predict(x), 2.5 * x**1.5)


def test_nn_fits_in_range():
    x, y = _toy()
    f = train_mlp(x, y, epochs=1500, seed=11)
    assert np.sqrt(np.mean((f.predict(x) - y) ** 2)) < 0.3 * np.std(y)


def test_power_law_exponent_is_measured_not_parsed():
    # the awkward form PySR actually emits
    e = power_law_exponent("f*0.6253*f**1.4784/f**(-1.1882)", "f", (1.0, 10.0))
    assert e is not None and abs(e - 11 / 3) < 1e-3
    # a non-power-law must be rejected, not fitted
    assert power_law_exponent("exp(-f) + 1.0", "f", (1.0, 10.0)) is None


def test_splits_do_not_overlap():
    itr, ite = split_random(40, 12, seed=11)
    assert len(itr) == 12 and len(set(itr) & set(ite)) == 0
    x = np.arange(40.0)
    a, b = split_extrapolate(x, 0.25)
    assert x[a].max() < x[b].min()


def test_problem_orients_x():
    p = Problem(
        track="t",
        x=np.arange(10.0),
        y=np.arange(10.0),
        law_np=law_np,
        law_t=law_t,
        params=[],
        theta_published={},
    )
    assert p.x.shape == (10, 1)


def _noise_problem(impl):
    x = np.linspace(1.0, 4.0, 20)
    return Problem(
        track="t",
        x=x,
        y=2.5 * x**1.5,
        law_np=law_np,
        law_t=law_t,
        params=[PhysParam("a", 1.0)],
        theta_published={"a": 2.5},
        sigma=np.full(20, 0.1),
        arm_impl={"physics": impl},
    )


def test_noise_sweep_widens_sigma_and_restores_the_problem():
    """The arms see the measurement sigma and the added noise in quadrature,
    and the problem comes back unchanged even when an arm raises."""
    from physprior.benchmark.protocol import sweep_noise

    seen = []

    def impl(prob, idx, seed, w_phys):
        seen.append(float(prob.sigma[0]))
        if len(seen) == 2:
            raise RuntimeError("arm failed")
        return oracle(law_np, {"a": 2.5})

    prob = _noise_problem(impl)
    y0, s0 = prob.y.copy(), prob.sigma.copy()
    with pytest.raises(RuntimeError):
        sweep_noise(prob, [0.0, 0.1], seeds=(11,), arms=("physics",), progress=False)
    assert seen[0] == pytest.approx(0.1)
    assert seen[1] == pytest.approx(np.hypot(0.1, 0.1 * np.std(y0)))
    assert np.array_equal(prob.y, y0) and np.array_equal(prob.sigma, s0)


def test_score_records_why_a_prediction_failed():
    from physprior.benchmark.protocol import score
    from physprior.methods.base import Fit

    def broken(xq):
        raise ValueError("no prediction")

    prob = _noise_problem(lambda *a: None)
    fit = Fit(name="physics", predict=broken)
    row = score(prob, fit, np.arange(10), np.arange(10, 20))
    assert row["rmse_in"] == np.inf
    assert "no prediction" in row["error"]


def test_pinn_reads_a_d_by_n_input_as_its_transpose():
    """A (d, N) input must reach the law as (N, d), the layout Standardiser
    uses, not as a reshape that mixes the columns."""
    rng = np.random.default_rng(0)
    x = rng.uniform(1.0, 2.0, (30, 2))
    y = 2.0 * x[:, 0] + x[:, 1]

    def lin_t(xx, a):
        return a * xx[:, 0] + xx[:, 1]

    kw = {"w_phys": 10.0, "epochs": 200, "seed": 11}
    a = fit_pinn(x, y, lin_t, [PhysParam("a", 1.0)], **kw)
    b = fit_pinn(x.T, y, lin_t, [PhysParam("a", 1.0)], **kw)
    assert a.params["a"] == pytest.approx(b.params["a"], rel=1e-10)
