"""Methods: each arm must recover a known answer on data where we know it."""

import numpy as np

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
