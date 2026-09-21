"""The protocol. Every track is run through exactly this, so the tracks can
be compared with each other and not only within themselves.

Six questions, asked the same way every time:

1  interpolation   fit on a random subset, test on the rest of the range
2  data efficiency the same, as the training budget shrinks
3  noise           the same, as noise is added to the targets
4  extrapolation   fit on the LOW part of the range, test on the HIGH part
5  parameters      does the arm return a physical constant, and is it right
6  law             does the arm return the functional form, and is it right

Arms are always: oracle, physics, pinn, sr, nn. The oracle is not a
competitor -- it is the ceiling, the published law with published constants,
and any arm that "beats" it in-sample is fitting noise.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from physprior.benchmark.metrics import mape, nrmse, rmse
from physprior.methods import neural as nn_mod
from physprior.methods import pinn as pinn_mod
from physprior.methods import symbolic as sr_mod
from physprior.methods.base import REPORT_SEEDS, Fit
from physprior.methods.pinn import PhysParam  # noqa: F401

F = TypeVar("F", bound=Callable[..., Any])

ARMS = ("oracle", "physics", "pinn", "sr", "nn")


def law(expression: str) -> Callable[[F], F]:
    """Attach a human-readable form to a law function.

    The closed form is carried on the callable itself so every arm that uses
    the law can report it without a parallel lookup table. Setting an
    attribute on a function is invisible to a type checker, so the single
    unavoidable ignore lives here rather than at each of the four call sites.
    """

    def decorate(fn: F) -> F:
        fn.expression = expression  # type: ignore[attr-defined]
        return fn

    return decorate


@dataclass
class Problem:
    """One track's data plus everything a physics-informed arm needs."""

    track: str
    x: np.ndarray  # (N, d) inputs, physical units
    y: np.ndarray  # (N,) target, physical units
    law_np: Callable  # law(x, **theta) -> y, numpy
    # None where the law is a differential equation rather than a closed
    # form -- `relativity/gw150914` supplies an ODE residual instead.
    law_t: Callable | None  # the same law in torch
    params: list  # list[PhysParam] to recover
    theta_published: dict[str, float]  # the literature values
    xlabel: str = "x"
    ylabel: str = "y"
    sigma: np.ndarray | None = None  # measurement 1-sigma, if the data has it
    sr_kwargs: dict = field(default_factory=dict)
    sr_transform: Callable | None = None  # (x, y) -> (X_sr, y_sr) if SR sees
    nn_cfg: dict = field(default_factory=dict)
    pinn_epochs: int = 6000
    # A track may replace an arm's implementation -- e.g. track G's `pinn` is
    # the ODE-residual PINN, because its law IS a differential equation.
    # Signature: impl(prob, idx, seed, w_phys) -> Fit
    arm_impl: dict = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        self.x = np.atleast_2d(np.asarray(self.x, float))
        if self.x.shape[0] != len(self.y):
            self.x = self.x.T
        self.y = np.asarray(self.y, float).ravel()

    def __len__(self) -> int:
        return len(self.y)

    def sub(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.x[idx], self.y[idx]


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def split_random(n: int, n_train: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.sort(perm[:n_train]), np.sort(perm[n_train:])


def split_extrapolate(
    x1d: np.ndarray, train_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    """Train on the low end of the range, test on the high end. No overlap."""
    order = np.argsort(x1d)
    k = max(round(train_frac * len(x1d)), 3)
    return np.sort(order[:k]), np.sort(order[k:])


# ---------------------------------------------------------------------------
# fitting one arm
# ---------------------------------------------------------------------------

# A symbolic search costs tens of seconds. The sweeps repeat it hundreds of
# times, so they run a shorter search; only the headline fits get the full
# budget. The shortened budget is recorded in every row that used it.
SR_SWEEP_ITERATIONS = 25


def fit_arm(
    arm: str,
    prob: Problem,
    idx: np.ndarray,
    seed: int,
    w_phys: float = 1.0,
    sr_seed: int | None = None,
    sr_fast: bool = False,
) -> Fit:
    if arm in prob.arm_impl:
        return prob.arm_impl[arm](prob, idx, seed, w_phys)

    xtr, ytr = prob.sub(idx)
    sig = prob.sigma[idx] if prob.sigma is not None else None

    if arm == "oracle":
        return pinn_mod.oracle(prob.law_np, prob.theta_published)

    if arm == "physics":
        return pinn_mod.fit_physics(xtr, ytr, prob.law_np, prob.params, sigma=sig)

    if arm == "pinn":
        f = pinn_mod.fit_pinn(
            xtr,
            ytr,
            prob.law_t,
            prob.params,
            w_phys=w_phys,
            seed=seed,
            epochs=prob.pinn_epochs,
        )
        # The arm IS the law plus a learned correction, so it does return a
        # closed form -- an incomplete one. Recording None here made the
        # capability table say "0/4 tracks", which is simply wrong.
        law = getattr(prob.law_np, "expression", None)
        f.expression = f"{law}  + NN(x)" if law else "law + NN(x)"
        return f

    if arm == "nn":
        cfg = dict(prob.nn_cfg)
        cfg.pop("tuned_val_rmse", None)
        return nn_mod.train_mlp(xtr, ytr, seed=seed, **cfg)

    if arm == "sr":
        Xs, ys, back = _sr_view(prob, xtr, ytr)
        kw = dict(prob.sr_kwargs)
        if sr_fast:
            kw["niterations"] = min(kw.get("niterations", 60), SR_SWEEP_ITERATIONS)
        f = sr_mod.fit_sr(Xs, ys, seed=sr_seed if sr_seed is not None else seed, **kw)
        if back is not None:
            raw_predict = f.predict

            def predict_in_original_units(xq, _p=raw_predict, _b=back):
                return _b(xq, _p)

            f.predict = predict_in_original_units
        return f

    raise ValueError(f"unknown arm {arm!r}")


def _sr_view(prob: Problem, x: np.ndarray, y: np.ndarray):
    """SR may see rescaled features. Returns (X, y, back-transform or None)."""
    if prob.sr_transform is None:
        return x, y, None
    return prob.sr_transform(x, y)


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def score(
    prob: Problem,
    fit: Fit,
    idx_in: np.ndarray,
    idx_out: np.ndarray,
    eval_y: np.ndarray | None = None,
    **tags,
) -> dict:
    """Score a fit.

    `eval_y` is the target to score AGAINST, which is not always the target
    that was fitted. In the noise sweep the arms are trained on corrupted
    values but evaluated against the clean measurements: the question there
    is who recovers the signal, not who reproduces the noise. Scoring against
    the corrupted values instead would drive every arm to the same noise
    floor and hide exactly the effect the sweep exists to measure.
    """
    row: dict[str, Any] = {"track": prob.track, "arm": fit.name, **tags}
    truth = prob.y if eval_y is None else np.asarray(eval_y, float).ravel()
    for label, idx in (("in", idx_in), ("out", idx_out)):
        if len(idx) == 0:
            row[f"rmse_{label}"] = np.nan
            row[f"nrmse_{label}"] = np.nan
            row[f"mape_{label}"] = np.nan
            continue
        xq, yq = prob.x[idx], truth[idx]
        try:
            pred = np.asarray(fit.predict(xq), float).ravel()
        except Exception:
            pred = np.full(len(yq), np.nan)
        row[f"rmse_{label}"] = rmse(yq, pred)
        row[f"nrmse_{label}"] = nrmse(yq, pred, scale=float(np.std(truth)))
        row[f"mape_{label}"] = mape(yq, pred)
    row["seconds"] = fit.seconds
    row["n_free"] = fit.n_free
    row["expression"] = fit.expression
    row["interpretable"] = fit.interpretable
    for k, v in fit.params.items():
        row[f"param_{k}"] = v
        if k in prob.theta_published:
            row[f"err_{k}_pct"] = (
                (v - prob.theta_published[k]) / prob.theta_published[k] * 100.0
            )
    for k, v in fit.param_sigma.items():
        row[f"sigma_{k}"] = v
    return row


# ---------------------------------------------------------------------------
# the three sweeps
# ---------------------------------------------------------------------------


def sweep_budget(
    prob: Problem,
    budgets,
    seeds=REPORT_SEEDS,
    arms=ARMS,
    w_phys: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    rows = []
    for nb in budgets:
        nb = int(min(nb, len(prob) - 2))
        for seed in seeds:
            itr, ite = split_random(len(prob), nb, seed)
            for arm in arms:
                f = fit_arm(arm, prob, itr, seed, w_phys=w_phys, sr_fast=True)
                rows.append(
                    score(prob, f, itr, ite, sweep="budget", n_train=nb, seed=seed)
                )
        if progress:
            print(f"  budget n_train={nb:4d} done", flush=True)
    return pd.DataFrame(rows)


def sweep_noise(
    prob: Problem,
    noise_fracs,
    seeds=REPORT_SEEDS,
    arms=ARMS,
    n_train: int | None = None,
    w_phys: float = 1.0,
    progress: bool = True,
) -> pd.DataFrame:
    """Noise is added as a fraction of the spread of y, on top of whatever
    measurement noise the real data already carries."""
    rows = []
    base_y = prob.y.copy()
    scale = float(np.std(base_y))
    n_train = n_train or max(int(0.7 * len(prob)), 4)
    for frac in noise_fracs:
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            prob.y = base_y + rng.normal(0.0, frac * scale, len(base_y))
            itr, ite = split_random(len(prob), n_train, seed)
            for arm in arms:
                f = fit_arm(arm, prob, itr, seed, w_phys=w_phys, sr_fast=True)
                rows.append(
                    score(
                        prob,
                        f,
                        itr,
                        ite,
                        eval_y=base_y,
                        sweep="noise",
                        noise_frac=frac,
                        seed=seed,
                        n_train=n_train,
                    )
                )
        if progress:
            print(f"  noise frac={frac:.3f} done", flush=True)
    prob.y = base_y
    return pd.DataFrame(rows)


def study_extrapolation(
    prob: Problem,
    train_frac: float = 0.5,
    seeds=REPORT_SEEDS,
    arms=ARMS,
    w_phys: float = 1.0,
) -> pd.DataFrame:
    itr, ite = split_extrapolate(prob.x[:, 0], train_frac)
    rows = []
    for seed in seeds:
        for arm in arms:
            f = fit_arm(arm, prob, itr, seed, w_phys=w_phys)
            rows.append(
                score(
                    prob,
                    f,
                    itr,
                    ite,
                    sweep="extrapolation",
                    train_frac=train_frac,
                    seed=seed,
                    n_train=len(itr),
                )
            )
    return pd.DataFrame(rows)


def sweep_physics_weight(
    prob: Problem, weights, seeds=REPORT_SEEDS, n_train: int | None = None
) -> pd.DataFrame:
    """The dial: how much is the physics term in the loss actually worth?"""
    rows = []
    n_train = n_train or max(int(0.7 * len(prob)), 4)
    for w in weights:
        for seed in seeds:
            itr, ite = split_random(len(prob), n_train, seed)
            f = fit_arm("pinn", prob, itr, seed, w_phys=w)
            rows.append(
                score(
                    prob,
                    f,
                    itr,
                    ite,
                    sweep="w_phys",
                    w_phys=w,
                    seed=seed,
                    n_train=n_train,
                )
            )
    return pd.DataFrame(rows)
