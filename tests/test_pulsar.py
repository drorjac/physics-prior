"""gravity/pulsar_spindown: the catalogue parser, the selection rule and the
simulated control. Everything here but the last test runs offline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physprior.data.sources import atnf
from physprior.problems.gravity import pulsar_spindown as S

RECORD = """#CATALOGUE 2.8.1
PSRJ     J0534+2200                    lgs+97
F0       29.946923                5    ljg+15
F1       -3.77535E-10             2    ljg+15
F2       1.1147E-20               5    ljg+15
TYPE     HE[cdt69]
@-----------------------------------------------------------------
PSRJ     J9999+0000                    xyz
F0       100.0                    1    xyz
F1       -1.0E-15                 1    xyz
F2       1.0E-30                  9    xyz
BINARY   BT
@-----------------------------------------------------------------
"""


def test_an_error_is_in_units_of_the_last_digit():
    assert atnf._value_and_error(["29.946923", "5"]) == pytest.approx((29.946923, 5e-6))
    v, e = atnf._value_and_error(["-3.77535E-10", "2"])
    assert v == pytest.approx(-3.77535e-10) and e == pytest.approx(2e-15)
    assert np.isnan(atnf._value_and_error(["1.0", "ljg+15"])[1])


def test_the_crab_gives_its_braking_index_and_the_binary_is_dropped():
    df = atnf.parse(RECORD)
    crab = df[df.psrj == "J0534+2200"].iloc[0]
    # n = nu nu_ddot / nu_dot^2 = 2.342 for these values
    assert crab.n_obs == pytest.approx(29.946923 * 1.1147e-20 / 3.77535e-10**2)
    assert 1200 < crab.age_yr < 1300
    assert 3.5e12 < crab.b_surface_g < 4.0e12
    assert list(atnf.select(df).psrj) == ["J0534+2200"]


def test_the_control_depends_on_age_as_stated():
    sim = S.simulated_problem(lo=np.array([2.8, 12.0]), hi=np.array([3.9, 14.0]))
    assert len(sim) == S.SIM_N
    slope = np.polyfit(sim.x[:, 0], sim.y, 1)[0]
    assert slope == pytest.approx(S.SIM_SLOPE, abs=0.1)


def test_the_verdicts_read_the_criteria_as_written():
    def ex(pinn, physics, out=2.0):
        rows = []
        for seed, (a, b) in zip((11, 23, 42), zip(pinn, physics), strict=True):
            rows += [
                {"seed": seed, "arm": "pinn", "nrmse_out": a},
                {"seed": seed, "arm": "physics", "nrmse_out": b},
                {"seed": seed, "arm": "nn", "nrmse_out": out},
            ]
        return pd.DataFrame(rows)

    dip = {"below_three": True}
    v = S.verdicts(ex([2, 2, 2], [2, 2, 2]), ex([0.3, 0.3, 0.3], [1, 1, 1]), dip)
    assert v["p2_pinn_not_better_on_every_seed"]
    assert v["p3_every_arm_fails_out_of_range"]
    assert v["p4_pinn_better_in_control_on_every_seed"]
    v = S.verdicts(ex([1, 1, 1], [2, 2, 2]), ex([1.2, 0.3, 0.3], [1, 1, 1]), dip)
    assert not v["p2_pinn_not_better_on_every_seed"]
    assert not v["p4_pinn_better_in_control_on_every_seed"]


@pytest.mark.network
def test_the_catalogue_selection():
    p = atnf.load()
    assert p.catalogue_version == "2.8.1"
    assert len(p) == 13
    young = p.frame.head(6)
    assert (young.n_obs.between(1.5, 3.0)).all()
