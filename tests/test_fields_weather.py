"""fields/weather: the ISD-Lite parser, the law, the splits, kriging and the
simulated control. Synthetic stations, so no download is needed."""

from __future__ import annotations

import numpy as np
import pytest

from physprior.benchmark.protocol import fit_arm
from physprior.data.sources import isd
from physprior.exceptions import UnitError
from physprior.problems.fields import kriging
from physprior.problems.fields import weather as W
from physprior.problems.fields import weather_sim as S

LINES = [
    "2023 07 01 12   215   120 10150   270    30     4     0 -9999",
    "2023 07 02 12 -9999   120 10150   270    30     4     0 -9999",
    "2023 07 03 12  -105   120 10150   270    30     4     0 -9999",
]


def _stations(n: int = 90, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.2, 3.0, n)
    lon = rng.uniform(6.0, 13.0, n)
    lat = rng.uniform(45.6, 47.9, n)
    return np.column_stack([z, lon, lat])


TRUE = {"T0": 18.0, "a": -0.4, "b": -0.9, "Gamma": -5.8}


def _problem(noise: float = 0.0, seed: int = 0):
    x = _stations(seed=seed)
    y = W.law_np(x, **TRUE) + np.random.default_rng(seed).normal(0, noise, len(x))
    return W.make_problem("fields/weather/test", x, y, {"Gamma": -6.5}, {})


# ---------------------------------------------------------------------------
# loader
# ---------------------------------------------------------------------------


def test_parse_lite_reads_tenths_and_missing():
    df = isd.parse_lite("\n".join(LINES), 2023)
    assert list(df.day) == [1, 2, 3]
    assert df.temp_c.iloc[0] == pytest.approx(21.5)
    assert np.isnan(df.temp_c.iloc[1])
    assert df.temp_c.iloc[2] == pytest.approx(-10.5)


def test_parse_lite_rejects_a_shifted_layout():
    shifted = [" " + ln for ln in LINES]
    with pytest.raises(UnitError):
        isd.parse_lite("\n".join(shifted), 2023)


def test_parse_lite_rejects_the_wrong_year():
    with pytest.raises(UnitError):
        isd.parse_lite("\n".join(LINES), 2022)


def test_snapshot_rule_text_states_the_rule():
    text = isd.SnapshotRule(year=2023, month=1).describe()
    assert "12 UTC" in text and "2023-01" in text and "80%" in text


# ---------------------------------------------------------------------------
# law, oracle, splits
# ---------------------------------------------------------------------------


def test_physics_recovers_the_constants_exactly_without_noise():
    prob = _problem()
    f = fit_arm("physics", prob, np.arange(len(prob)), seed=0)
    for k, v in TRUE.items():
        assert f.params[k] == pytest.approx(v, abs=1e-6)
    assert W.effective_lapse_rate(f, prob.x) == pytest.approx(TRUE["Gamma"], abs=1e-8)


def test_oracle_holds_gamma_at_the_published_value():
    prob = _problem()
    f = fit_arm("oracle", prob, np.arange(len(prob)), seed=0)
    assert f.params["Gamma"] == W.GAMMA_STD
    assert f.n_free == 3


def test_elevation_split_puts_every_test_station_above_the_training_ones():
    prob = _problem()
    itr, ite = W.elevation_split(prob)
    assert prob.x[ite, 0].min() >= prob.x[itr, 0].max()
    assert len(itr) == round(W.ELEV_TRAIN_FRAC * len(prob))


def test_blocks_partition_the_stations():
    prob = _problem()
    held = np.concatenate([ite for _, _, ite in W.block_splits(prob)])
    assert sorted(held) == list(range(len(prob)))
    for _, itr, ite in W.block_splits(prob):
        assert not set(itr) & set(ite)


def test_lapse_rate_derivative_is_step_independent_for_the_law():
    prob = _problem()
    f = fit_arm("physics", prob, np.arange(len(prob)), seed=0)
    rows = W.lapse_step_study(f, prob.x)
    assert all(abs(r["dTdz"] - TRUE["Gamma"]) < 1e-8 for r in rows)


# ---------------------------------------------------------------------------
# kriging
# ---------------------------------------------------------------------------


def test_universal_kriging_recovers_gamma_from_a_noisy_linear_field():
    prob = _problem(noise=0.2)
    f = W.fit_kriging(prob, np.arange(len(prob)), "uk")
    assert f.params["Gamma"] == pytest.approx(
        TRUE["Gamma"], abs=4 * f.param_sigma["Gamma"]
    )
    assert f.param_sigma["Gamma"] < 0.5


def test_kriging_interpolates_a_smooth_field():
    x = _stations(120, seed=1)
    X = kriging.to_km(x, W.LON0, W.LAT0)
    y = np.sin(X[:, 0] / 150.0) + 0.5 * x[:, 0]
    tr, te = np.arange(100), np.arange(100, 120)
    f = kriging.fit_gp(x[tr], y[tr], W.const_basis, lon0=W.LON0, lat0=W.LAT0)
    err = np.sqrt(np.mean((f.predict(x[te]) - y[te]) ** 2))
    assert err < 0.2 * np.std(y)


def test_kriging_refuses_heights_in_metres():
    x = _stations(20)
    x[:, 0] *= 1000.0
    with pytest.raises(UnitError):
        kriging.fit_gp(x, x[:, 0], W.const_basis, lon0=W.LON0, lat0=W.LAT0)


# ---------------------------------------------------------------------------
# simulated control
# ---------------------------------------------------------------------------


def test_simulation_is_reproducible_and_has_the_stated_parts():
    x = _stations()
    a = S.simulate(x, "law", seed=5)
    b = S.simulate(x, "law", seed=5)
    np.testing.assert_array_equal(a["y"], b["y"])
    np.testing.assert_allclose(a["y"], a["mean"] + a["residual"] + a["noise"])


def test_best_linear_gamma_is_the_truth_without_the_cold_pool():
    x = _stations()
    assert S.best_linear_gamma(x, "law") == pytest.approx(S.TRUTH["Gamma"])
    # a cold pool at the bottom makes the fitted lapse rate shallower
    assert S.best_linear_gamma(x, "inversion") > S.TRUTH["Gamma"]


def test_cold_pool_vanishes_above_its_top():
    z = np.array([0.0, 0.6, 1.2, 2.0])
    np.testing.assert_allclose(S.cold_pool(z, 6.0, 1.2), [6.0, 3.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def test_run_lists_the_weather_subrun():
    from physprior.problems.fields import run

    assert ("weather", "physprior.problems.fields.weather") in run.SUBRUNS


def test_tutorial_builds():
    from physprior.reporting.tutorials_fields import t10_spatial_fields

    nb = t10_spatial_fields()
    assert len(nb.cells) > 10


@pytest.mark.network
@pytest.mark.slow
def test_real_snapshot_meets_its_promises():
    fld = W.load_field("july")
    assert len(fld) >= 100
    assert fld.elev_m.max() > 3000
    assert np.all(fld.n_days >= 0.8 * 31)


def test_pinn_constants_start_from_training_statistics_not_the_answer():
    y = np.array([20.0, 22.0, 24.0])
    ps = {p.name: p for p in W.pinn_params(y)}
    assert ps["T0"].init == pytest.approx(22.0)
    # multiplicative, negative, and not the published value
    assert ps["Gamma"].positive and ps["Gamma"].init == -1.0
    assert "pinn" in W.make_problem("t", _stations(), np.zeros(90), {}, {}).arm_impl
