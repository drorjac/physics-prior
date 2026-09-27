"""The `pinn` arm's configuration: no loss balancing, a tuned weight per track.

The selection rule is tested on its own, with the fits replaced by a known
validation curve, so these run in CI in milliseconds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from physprior.benchmark import pinn_tuning as T
from physprior.benchmark.protocol import Problem, fit_arm
from physprior.methods import pinn as P


def _problem(n: int = 20, w: float | None = None) -> Problem:
    x = np.linspace(1.0, 2.0, n)
    return Problem(
        track="test/track",
        x=x.reshape(-1, 1),
        y=2.0 * x,
        law_np=lambda x, a: a * np.asarray(x)[:, 0],
        law_t=lambda x, a: a * x[:, 0],
        params=[P.PhysParam("a", 1.0, lo=0.1, hi=10.0)],
        theta_published={"a": 2.0},
        pinn_epochs=50,
        pinn_w_phys=w,
    )


def test_the_frozen_default_does_not_anneal_the_weight():
    assert P.FROZEN_PINN is P.DEFAULT_PINN
    assert not P.FROZEN_PINN.balance


def test_the_validation_block_is_the_top_of_the_training_range():
    prob = _problem()
    train = np.arange(16)
    fit, val = T.inner_split(prob, train)
    assert len(val) == 4 and prob.x[val, 0].min() > prob.x[fit, 0].max()
    assert set(fit) | set(val) == set(train)


def _curve(scores: dict[float, float]):
    def fake(prob, fit_idx, val_idx, scale, grid, seeds):
        return pd.DataFrame(
            [
                {"w_phys": w, "seed": s, "val_nrmse": scores[w]}
                for w in grid
                for s in seeds
            ]
        )

    return fake


def test_ties_go_to_the_larger_weight(monkeypatch):
    scores = dict.fromkeys(T.W_GRID, 1.0) | {1.0: 0.5, 10.0: 0.505}
    monkeypatch.setattr(T, "_score_grid", _curve(scores))
    chosen, df = T.select_w_phys(_problem(), np.arange(16))
    assert chosen == 10.0
    assert not df.pinned_at_edge.iloc[0]


def test_an_edge_choice_extends_the_grid_until_it_is_interior(monkeypatch):
    scores = {w: 1.0 / w for w in (*T.W_GRID, 1e4, 1e5)} | {1e6: 1.0}
    monkeypatch.setattr(T, "_score_grid", _curve(scores))
    chosen, df = T.select_w_phys(_problem(), np.arange(16))
    assert chosen == 1e5
    assert not df.pinned_at_edge.iloc[0]
    assert df.w_phys.max() == 1e6


def test_a_choice_still_at_the_edge_is_reported_as_pinned(monkeypatch):
    grid = [w * 10.0**k for w in T.W_GRID for k in range(4)]
    monkeypatch.setattr(T, "_score_grid", _curve({w: 1.0 / w for w in grid}))
    chosen, df = T.select_w_phys(_problem(), np.arange(16))
    assert chosen == max(T.W_GRID) * 10.0**T.MAX_EXTEND
    assert df.pinned_at_edge.iloc[0]


def test_fit_arm_uses_the_tracks_weight_unless_told_otherwise():
    idx = np.arange(20)
    tuned = fit_arm("pinn", _problem(w=37.0), idx, seed=3)
    assert tuned.extra["w_phys"] == pytest.approx(37.0)
    explicit = fit_arm("pinn", _problem(w=37.0), idx, seed=3, w_phys=2.0)
    assert explicit.extra["w_phys"] == pytest.approx(2.0)
    default = fit_arm("pinn", _problem(), idx, seed=3)
    assert default.extra["w_phys"] == pytest.approx(1.0)
