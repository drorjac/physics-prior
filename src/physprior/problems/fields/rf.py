"""Track fields/rf -- where is the transmitter, and what is the field?

DATA     Simulated. A 2.4 GHz transmitter in a 24 x 18 wavelength floor plan
         (3 x 2.25 m), solved exactly by the 2-D Helmholtz equation
         (`rf_sim`). Receivers at random positions read the local-mean power
         in dB with Gaussian noise of NOISE_DB (log-normal shadowing in
         linear units). The true field is known everywhere, so every arm is
         scored against the map it was trying to reconstruct, not only
         against held-out noisy readings.

         Two scenes, the controlled comparison:
           free    no walls. The log-distance law is exact (n = 1 in 2-D).
           walls   three rooms of concrete walls with doorways and a pillar.
                   The law knows distance and nothing about walls.

LAW      Log-distance path loss, the received-power form:

             P(x, y) = P0 - 10 n log10( d / d0 ),   d = |(x, y) - (xt, yt)|

         with d0 = 1 wavelength.

PARAM    P0, n, and the transmitter position (xt, yt) -- an inverse problem:
         the arm must say where the source is.

ORACLE   The same law with the TRUE transmitter position, n = 1 and the P0 of
         the analytic free-space field (the large-r limit of the Hankel
         function, `rf_sim.free_space_law_db`). It is the ceiling of the LAW,
         not of the problem: in the walls scene it is wrong by the wall
         losses, and arms that learn those losses from data can beat it.
         The true simulated field is not used as an arm -- it is the target.

ARMS     oracle, physics (law fitted, multi-start over the transmitter
         position), pinn (law + learned correction, shape B, started from the
         physics fit), nn (tuned MLP), gp (ordinary kriging: constant mean +
         GP), kriging (the fitted law as the mean + GP on its residuals),
         and pinn_free (the pinn with w_phys = 1 fixed, no loss balancing --
         a diagnostic, the arm's fixed-weight reference beside the per-scene
         tuned weight). A Helmholtz-residual PINN
         (shape A) is run separately on a window of the scene
         (`helmholtz_window_study`), since it needs the refractive-index map
         and a complex field the readings do not contain.

H3       The free-space control is the case where the law is complete: a
         correction can only add noise. In the walls scene the missing
         physics is a step loss at each wall, a shape the smooth law cannot
         make: H3 predicts a correction should help there.

EXTRAPOLATION   Train on receivers in the transmitter's room (x < 9), predict
         the other rooms, behind one and two walls.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from physprior.benchmark.metrics import rmse
from physprior.benchmark.protocol import (
    Problem,
    fit_arm,
    law,
    score,
    split_extrapolate,
    split_random,
)
from physprior.config import get_settings
from physprior.io import save_json, save_table
from physprior.methods import pinn as pinn_mod
from physprior.methods.base import REPORT_SEEDS, TUNE_SEEDS, Fit
from physprior.methods.pinn import PhysParam
from physprior.units import require, require_range

from . import rf_sim as S

TRACK = "fields/rf"  # results/fields/rf/...
FIG = "fields"  # figures/fields/rf_*.png

NOISE_DB = 2.0  # receiver noise, 1 sigma in dB
N_POOL = 600  # candidate receiver positions per scene
DATA_SEED = 2026  # draws the receiver positions and the noise; not a model seed
TX_EXCLUSION = 1.5  # lambda; no receiver this close to the transmitter
D_SOFT = 0.1  # lambda; softens log10(d) at d -> 0 so gradients stay finite
EVAL_STEP = 0.25  # lambda; the lattice on which maps are scored
ROOM_EDGE = 9.0  # x < 9 is the transmitter's room (wall 1 is x = 9..10)
BEYOND = 10.0  # x > 10 is behind at least one wall

ARMS = ("oracle", "physics", "pinn", "pinn_free", "nn", "gp", "kriging")

# The `pinn` arm's physics weight per scene, chosen on the tuning seeds by
# `physprior.problems.fields.retune` (results/fields/rf_<scene>/tune/). Both
# are pinned at the top of the extended grid: the validation block prefers
# no correction, so the arm is the law by choice.
PINN_W_PHYS = {"free": 1e6, "walls": 1e6}
EPOCHS = 4000  # nn and pinn alike

# Fallback when a run is too quick to afford the convergence study.
PPW_QUICK = 8


# ---------------------------------------------------------------------------
# the law
# ---------------------------------------------------------------------------


@law("P0 - 10 n log10(|x - x_t| / d0)")
def law_np(x, P0, n, xt, yt):
    x = np.atleast_2d(np.asarray(x, float))
    d = np.sqrt((x[:, 0] - xt) ** 2 + (x[:, 1] - yt) ** 2 + D_SOFT**2)
    return P0 - 10.0 * n * np.log10(d)


def law_t(x, P0, n, xt, yt):
    import torch

    d = torch.sqrt((x[:, 0] - xt) ** 2 + (x[:, 1] - yt) ** 2 + D_SOFT**2)
    return P0 - 10.0 * n * torch.log10(d)


def _params(scene: S.Scene, init: dict[str, float] | None = None) -> list[PhysParam]:
    init = init or {
        "P0": -30.0,
        "n": 1.5,
        "xt": scene.width / 2,
        "yt": scene.height / 2,
    }
    return [
        PhysParam("P0", init["P0"], positive=False, lo=-120.0, hi=40.0),
        PhysParam("n", init["n"], positive=True, lo=0.2, hi=8.0),
        PhysParam("xt", init["xt"], positive=False, lo=-4.0, hi=scene.width + 4.0),
        PhysParam("yt", init["yt"], positive=False, lo=-4.0, hi=scene.height + 4.0),
    ]


# ---------------------------------------------------------------------------
# the data
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    """One solved scene with its receivers and its scoring lattice."""

    name: str
    field: S.Field
    local_mean: np.ndarray  # (ny, nx) dB
    pool_x: np.ndarray  # (N, 2) receiver positions, lambda
    pool_truth: np.ndarray  # (N,) local-mean dB at the receivers
    pool_noisy: np.ndarray  # (N,) what the receivers read
    eval_x: np.ndarray  # (M, 2) scoring lattice
    eval_truth: np.ndarray  # (M,)

    @property
    def scene(self) -> S.Scene:
        return self.field.scene


def _valid(scene: S.Scene, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    r = np.hypot(X - scene.tx[0], Y - scene.tx[1])
    m = S.AVG_RADIUS
    inside = (np.abs(X - scene.width / 2) <= scene.width / 2 - m) & (
        np.abs(Y - scene.height / 2) <= scene.height / 2 - m
    )
    # receivers stand in air: the walls of the floor plan are excluded in BOTH
    # scenes, so the free-space control is scored on exactly the same points
    walls = S.floor_plan().in_material(X, Y)
    return inside & ~walls & (r >= TX_EXCLUSION)


def build_scenario(
    name: str,
    ppw: int,
    n_pool: int = N_POOL,
    noise_db: float = NOISE_DB,
    seed: int = DATA_SEED,
) -> Scenario:
    scene = S.SCENES[name]()
    fld = S.solve(scene, ppw)
    lm = fld.local_mean_db()
    X, Y = np.meshgrid(fld.x, fld.y)
    ok = _valid(scene, X, Y)
    rng = np.random.default_rng(seed)
    flat = np.flatnonzero(ok.ravel())
    pick = rng.choice(flat, size=n_pool, replace=False)
    px, py = X.ravel()[pick], Y.ravel()[pick]
    truth = lm.ravel()[pick]
    noisy = truth + rng.normal(0.0, noise_db, n_pool)

    ex = np.arange(S.AVG_RADIUS, scene.width - S.AVG_RADIUS + 1e-9, EVAL_STEP)
    ey = np.arange(S.AVG_RADIUS, scene.height - S.AVG_RADIUS + 1e-9, EVAL_STEP)
    EX, EY = np.meshgrid(ex, ey)
    keep = _valid(scene, EX, EY)
    eval_x = np.column_stack([EX[keep], EY[keep]])
    eval_truth = fld.at(lm, eval_x[:, 0], eval_x[:, 1])

    # dB of |u|^2 for a unit source: tens of dB below zero, never positive
    require_range(float(truth.max()), -80.0, 0.0, f"{name}: peak received power", "dB")
    require(bool(np.all(np.isfinite(eval_truth))), f"{name}: non-finite truth map")
    return Scenario(
        name=name,
        field=fld,
        local_mean=lm,
        pool_x=np.column_stack([px, py]),
        pool_truth=truth,
        pool_noisy=noisy,
        eval_x=eval_x,
        eval_truth=eval_truth,
    )


# ---------------------------------------------------------------------------
# arms that are not the protocol's defaults
# ---------------------------------------------------------------------------


def _starts(scene: S.Scene, x: np.ndarray, y: np.ndarray) -> list[dict[str, float]]:
    """Multi-start for the transmitter position: a 3 x 2 lattice over the
    scene, plus the strongest reading. The law is not convex in (xt, yt)."""
    out = []
    top = int(np.argmax(y))
    cand = [(x[top, 0], x[top, 1])] + [
        (fx * scene.width, fy * scene.height)
        for fx in (1 / 6, 1 / 2, 5 / 6)
        for fy in (1 / 4, 3 / 4)
    ]
    for cx, cy in cand:
        out.append({"P0": float(np.max(y)), "n": 1.5, "xt": float(cx), "yt": float(cy)})
    return out


def fit_law(scene: S.Scene, x: np.ndarray, y: np.ndarray, name: str = "physics") -> Fit:
    """Least squares on the law from every start; the lowest residual wins."""
    t0 = time.time()
    best, best_sse = None, np.inf
    for init in _starts(scene, x, y):
        try:
            f = pinn_mod.fit_physics(x, y, law_np, _params(scene, init), name=name)
        except (RuntimeError, ValueError):
            continue
        sse = float(np.sum((f.predict(x) - y) ** 2))
        if sse < best_sse:
            best, best_sse = f, sse
    if best is None:
        raise RuntimeError("the log-distance fit failed from every start")
    best.seconds = time.time() - t0
    lo = {p.name: p.lo for p in _params(scene)}
    hi = {p.name: p.hi for p in _params(scene)}
    best.extra["at_bound"] = {
        k: bool(min(abs(v - lo[k]), abs(v - hi[k])) < 1e-3 * (hi[k] - lo[k]))
        for k, v in best.params.items()
    }
    return best


def _gp(x: np.ndarray, r: np.ndarray, seed: int):
    """ML-II Gaussian process on (x, y) in wavelengths. Returns the model."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

    kern = ConstantKernel(10.0, (1e-2, 1e4)) * RBF(2.0, (0.2, 50.0)) + WhiteKernel(
        NOISE_DB**2, (1e-2, 1e3)
    )
    gp = GaussianProcessRegressor(
        kern, normalize_y=True, n_restarts_optimizer=2, random_state=seed
    )
    gp.fit(x, r)
    return gp


def _gp_extra(gp) -> dict:
    k = gp.kernel_
    ls = float(k.k1.k2.length_scale)
    return {
        "length_scale": ls,
        "noise_var": float(k.k2.noise_level),
        "length_at_bound": bool(ls < 0.21 or ls > 49.0),
    }


def make_arms(scene: S.Scene) -> dict:
    """arm_impl entries. Signature: impl(prob, idx, seed, w_phys) -> Fit."""

    def physics(prob, idx, seed, w_phys):
        return fit_law(scene, prob.x[idx], prob.y[idx])

    def _pinn(prob, idx, seed, w_phys, options, name):
        x, y = prob.x[idx], prob.y[idx]
        start = fit_law(scene, x, y)
        f = pinn_mod.fit_pinn(
            x,
            y,
            law_t,
            _params(scene, start.params),
            w_phys=w_phys,
            seed=seed,
            epochs=prob.pinn_epochs,
            options=options,
            name=name,
        )
        f.expression = f"{law_np.expression}  + NN(x)"
        f.seconds += start.seconds
        return f

    def pinn(prob, idx, seed, w_phys, options=pinn_mod.FROZEN_PINN):
        return _pinn(prob, idx, seed, w_phys, options, "pinn")

    def pinn_free(prob, idx, seed, w_phys):
        return _pinn(prob, idx, seed, 1.0, pinn_mod.DEFAULT_PINN, "pinn_free")

    def gp(prob, idx, seed, w_phys):
        t0 = time.time()
        x, y = prob.x[idx], prob.y[idx]
        model = _gp(x, y, seed)
        return Fit(
            name="gp",
            predict=lambda xq: model.predict(np.asarray(xq, float)),
            n_free=3,
            seconds=time.time() - t0,
            extra=_gp_extra(model),
        )

    def kriging(prob, idx, seed, w_phys):
        t0 = time.time()
        x, y = prob.x[idx], prob.y[idx]
        base = fit_law(scene, x, y, name="kriging")
        model = _gp(x, y - base.predict(x), seed)
        return Fit(
            name="kriging",
            predict=lambda xq: base.predict(xq) + model.predict(np.asarray(xq, float)),
            params=dict(base.params),
            param_sigma=dict(base.param_sigma),
            expression=f"{law_np.expression}  + GP(x)",
            n_free=4 + 3,
            seconds=time.time() - t0,
            extra=_gp_extra(model),
        )

    return {
        "physics": physics,
        "pinn": pinn,
        "pinn_free": pinn_free,
        "gp": gp,
        "kriging": kriging,
    }


def make_problem(sc: Scenario, nn_cfg: dict | None = None) -> Problem:
    return Problem(
        track=f"fields/rf_{sc.name}",
        x=sc.pool_x,
        y=sc.pool_noisy.copy(),
        law_np=law_np,
        law_t=law_t,
        params=_params(sc.scene),
        theta_published={
            "P0": S.P0_FREE_SPACE,
            "n": S.N_FREE_SPACE,
            "xt": sc.scene.tx[0],
            "yt": sc.scene.tx[1],
        },
        xlabel="x  [wavelengths]",
        ylabel="received power  [dB]",
        nn_cfg=dict(nn_cfg or {"width": 32, "depth": 3, "weight_decay": 1e-4}),
        pinn_epochs=EPOCHS,
        pinn_w_phys=PINN_W_PHYS[sc.name],
        arm_impl=make_arms(sc.scene),
        notes=f"simulated, 2-D Helmholtz, scene={sc.name}, noise {NOISE_DB} dB",
    )


def tune_nn(sc: Scenario, epochs: int = EPOCHS) -> dict:
    """The MLP's configuration, on the TUNING seeds, on a random half of the
    extrapolation training split (the transmitter's room).

    The receivers beyond the room are scored by the extrapolation study, so
    they take no part in the choice. Validation is against the noisy
    readings, as it would have to be with real receivers; the truth map is
    never consulted.
    """
    from physprior.methods.neural import tune_mlp

    itr, _ = extrapolation_split(make_problem(sc))
    a_idx, b_idx = split_random(len(itr), len(itr) // 2, seed=TUNE_SEEDS[0])
    a, b = itr[a_idx], itr[b_idx]
    return tune_mlp(
        sc.pool_x[a],
        sc.pool_noisy[a],
        sc.pool_x[b],
        sc.pool_noisy[b],
        TUNE_SEEDS,
        epochs,
    )


# ---------------------------------------------------------------------------
# scoring against the true map
# ---------------------------------------------------------------------------


def tx_error(fit: Fit, sc: Scenario) -> float:
    if "xt" not in fit.params:
        return float("nan")
    return float(
        np.hypot(fit.params["xt"] - sc.scene.tx[0], fit.params["yt"] - sc.scene.tx[1])
    )


def map_scores(fit: Fit, sc: Scenario) -> dict[str, Any]:
    """RMSE in dB against the true local-mean map, overall and by region."""
    pred = np.asarray(fit.predict(sc.eval_x), float).ravel()
    ex = sc.eval_x[:, 0]
    room, beyond = ex < ROOM_EDGE, ex > BEYOND
    return {
        "map_rmse_db": rmse(sc.eval_truth, pred),
        "map_rmse_room_db": rmse(sc.eval_truth[room], pred[room]),
        "map_rmse_beyond_db": rmse(sc.eval_truth[beyond], pred[beyond]),
        "tx_err_lambda": tx_error(fit, sc),
    }


def _row(prob, sc, fit, itr, ite, eval_y=None, **tags) -> dict:
    """The protocol's row (held-out readings) plus the map-level scores."""
    row = score(prob, fit, itr, ite, eval_y=eval_y, **tags)
    row.update(map_scores(fit, sc))
    row["scene"] = sc.name
    for k in ("correction_rms_frac", "w_phys_final", "length_scale", "length_at_bound"):
        if k in fit.extra:
            row[k] = fit.extra[k]
    if "at_bound" in fit.extra:
        row["physics_at_bound"] = any(fit.extra["at_bound"].values())
    return row


# ---------------------------------------------------------------------------
# the sweeps, on the protocol's own splits
# ---------------------------------------------------------------------------


def sweep_budget(prob, sc, budgets, arms=ARMS, seeds=REPORT_SEEDS) -> pd.DataFrame:
    """`protocol.sweep_budget`, with the map scored as well as the readings."""
    rows = []
    for nb in budgets:
        for seed in seeds:
            itr, ite = split_random(len(prob), int(nb), seed)
            for arm in arms:
                f = fit_arm(arm, prob, itr, seed)
                rows.append(
                    _row(
                        prob,
                        sc,
                        f,
                        itr,
                        ite,
                        sweep="budget",
                        n_train=int(nb),
                        seed=seed,
                    )
                )
        print(f"   [{sc.name}] budget {nb} done", flush=True)
    return pd.DataFrame(rows)


def sweep_noise(
    prob, sc, noise_dbs, n_train: int, arms=ARMS, seeds=REPORT_SEEDS
) -> pd.DataFrame:
    """Readings redrawn at each noise level, scored against the CLEAN field.

    Noise is set in dB rather than as a fraction of the spread of y, because
    receiver noise is quoted in dB.
    """
    rows = []
    base = prob.y.copy()
    try:
        for s_db in noise_dbs:
            for seed in seeds:
                rng = np.random.default_rng(1000 + seed)
                prob.y = sc.pool_truth + rng.normal(0.0, s_db, len(base))
                itr, ite = split_random(len(prob), n_train, seed)
                for arm in arms:
                    f = fit_arm(arm, prob, itr, seed)
                    rows.append(
                        _row(
                            prob,
                            sc,
                            f,
                            itr,
                            ite,
                            eval_y=sc.pool_truth,
                            sweep="noise",
                            noise_db=s_db,
                            n_train=n_train,
                            seed=seed,
                        )
                    )
            print(f"   [{sc.name}] noise {s_db} dB done", flush=True)
    finally:
        prob.y = base
    return pd.DataFrame(rows)


def extrapolation_split(prob) -> tuple[np.ndarray, np.ndarray]:
    """Transmitter's room (x < 9) against the rest, via the protocol split."""
    frac = float(np.mean(prob.x[:, 0] < ROOM_EDGE))
    itr, ite = split_extrapolate(prob.x[:, 0], frac)
    require(
        bool(prob.x[itr, 0].max() < ROOM_EDGE < prob.x[ite, 0].min()),
        "extrapolation split must be the transmitter's room against the rest",
    )
    return itr, ite


def study_extrapolation(prob, sc, arms=ARMS, seeds=REPORT_SEEDS) -> pd.DataFrame:
    itr, ite = extrapolation_split(prob)
    rows = []
    for seed in seeds:
        for arm in arms:
            f = fit_arm(arm, prob, itr, seed)
            rows.append(
                _row(
                    prob,
                    sc,
                    f,
                    itr,
                    ite,
                    sweep="extrapolation",
                    n_train=len(itr),
                    seed=seed,
                )
            )
    print(f"   [{sc.name}] extrapolation done", flush=True)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# shape A: a Helmholtz-residual PINN on a window of the scene
# ---------------------------------------------------------------------------

WINDOW = (11.0, 19.0, 6.0, 14.0)  # x0, x1, y0, y1 in lambda; excludes the source
HELMHOLTZ_EPOCHS = 1500


def fit_helmholtz_pinn(
    sc: Scenario,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    epochs: int = 3000,
    n_colloc: int = 768,
    n_ring: int = 8,
    w_pde: float = 1.0,
    n_fourier: int = 48,
    width: int = 48,
    depth: int = 3,
    lr: float = 2e-3,
) -> Fit:
    """A network u(x, y) = Re + i Im, trained on

        L = mean((P_net - y)^2) / NOISE_DB^2
            + w_pde mean(|lap u + k^2 n(x)^2 u|^2) / (k^2 u_scale)^2

    P_net is 10 log10 of |u|^2 averaged over `n_ring` points of the receiver's
    averaging disk, so the network is compared with what a receiver reads.
    n(x) is the TRUE refractive-index map: this arm is told the geometry.
    The readings carry no phase, and a window with no boundary condition
    admits many Helmholtz solutions, so the problem is under-determined; the
    study measures how far the residual narrows it. w_pde = 1 is not tuned.
    """
    import torch

    from physprior.methods.base import set_seed
    from physprior.methods.neural import DTYPE, mlp
    from physprior.methods.pinn import FourierFeatures

    t0 = time.time()
    set_seed(seed)
    x0, x1, y0, y1 = WINDOW
    centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    half = np.array([(x1 - x0) / 2, (y1 - y0) / 2])
    u_scale = float(10 ** (np.mean(y) / 20.0))

    gen = torch.Generator().manual_seed(seed)
    # one cycle per wavelength is the field's own scale
    ff = FourierFeatures(2, n_fourier, 1.0, gen)
    body = mlp(2 * n_fourier, width, depth, d_out=2)

    def net(p):  # p in lambda
        return body(ff(p))

    # receivers: points on the averaging disk (centre + two rings)
    ang = np.linspace(0.0, 2 * np.pi, n_ring, endpoint=False)
    offs = [np.zeros((1, 2))]
    for rr in (0.5 * S.AVG_RADIUS, 0.9 * S.AVG_RADIUS):
        offs.append(rr * np.column_stack([np.cos(ang), np.sin(ang)]))
    off = np.concatenate(offs)
    pts = (x[:, None, :] + off[None, :, :]).reshape(-1, 2)
    pts_t = torch.tensor(pts, dtype=DTYPE)
    y_t = torch.tensor(y, dtype=DTYPE)

    rng = np.random.default_rng(seed)
    col = centre + half * rng.uniform(-1, 1, (n_colloc, 2))
    eps = sc.scene.eps(col[:, 0], col[:, 1])
    col_t = torch.tensor(col, dtype=DTYPE, requires_grad=True)
    er = torch.tensor(eps.real, dtype=DTYPE)
    ei = torch.tensor(eps.imag, dtype=DTYPE)
    k2 = S.K0**2

    opt = torch.optim.Adam(list(body.parameters()), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def power_db(p):
        uv = net(p) * u_scale
        pw = (uv**2).sum(-1).reshape(len(y), -1).mean(-1)
        return 10.0 * torch.log10(pw + 1e-30)

    def laplacian(out, p):
        g = torch.autograd.grad(out.sum(), p, create_graph=True)[0]
        gxx = torch.autograd.grad(g[:, 0].sum(), p, create_graph=True)[0][:, 0]
        gyy = torch.autograd.grad(g[:, 1].sum(), p, create_graph=True)[0][:, 1]
        return gxx + gyy

    hist: dict[str, list[float]] = {"data": [], "pde": []}
    for _ep in range(epochs):
        opt.zero_grad()
        data = torch.mean((power_db(pts_t) - y_t) ** 2) / NOISE_DB**2
        uv = net(col_t)
        ur, ui = uv[:, 0], uv[:, 1]
        # (er + i ei)(ur + i ui) = (er ur - ei ui) + i (er ui + ei ur)
        rr = laplacian(ur, col_t) + k2 * (er * ur - ei * ui)
        ri = laplacian(ui, col_t) + k2 * (er * ui + ei * ur)
        pde = torch.mean(rr**2 + ri**2) / k2**2
        (data + w_pde * pde).backward()
        opt.step()
        sched.step()
        if _ep % 100 == 0 or _ep == epochs - 1:
            hist["data"].append(float(data.detach()))
            hist["pde"].append(float(pde.detach()))

    def predict(xq):
        xq = np.asarray(xq, float)
        out = []
        for chunk in np.array_split(xq, max(1, len(xq) // 2000)):
            q = (chunk[:, None, :] + off[None, :, :]).reshape(-1, 2)
            with torch.no_grad():
                uv = net(torch.tensor(q, dtype=DTYPE)) * u_scale
                pw = (uv**2).sum(-1).reshape(len(chunk), -1).mean(-1)
            out.append(10.0 * np.log10(pw.numpy() + 1e-30))
        return np.concatenate(out)

    return Fit(
        name="pinn_helmholtz",
        predict=predict,
        n_free=sum(p.numel() for p in body.parameters()),
        seconds=time.time() - t0,
        expression=None,
        extra={
            "final_data_loss": hist["data"][-1],
            "final_pde_loss": hist["pde"][-1],
            "w_pde": w_pde,
            "epochs": epochs,
            "history": hist,
        },
    )


def in_window(xy: np.ndarray) -> np.ndarray:
    x0, x1, y0, y1 = WINDOW
    return (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)


def helmholtz_window_study(
    sc: Scenario, nn_cfg: dict, seeds=REPORT_SEEDS, epochs: int = 3000
) -> tuple[pd.DataFrame, dict]:
    """Every receiver in the window trains; the window's map is scored.

    The comparison arms see the same readings. The physics arm has to place
    a transmitter that lies OUTSIDE the window from readings inside it.
    """
    from physprior.methods.neural import train_mlp

    wmask = in_window(sc.pool_x)
    x, y = sc.pool_x[wmask], sc.pool_noisy[wmask]
    emask = in_window(sc.eval_x)
    ex, et = sc.eval_x[emask], sc.eval_truth[emask]
    rows, keep = [], {}
    for seed in seeds:
        fits: dict[str, Fit] = {
            "physics": fit_law(sc.scene, x, y),
            "nn": train_mlp(x, y, seed=seed, epochs=EPOCHS, **nn_cfg),
        }
        gpm = _gp(x, y, seed)
        fits["gp"] = Fit("gp", predict=gpm.predict)
        fits["pinn_helmholtz"] = fit_helmholtz_pinn(sc, x, y, seed=seed, epochs=epochs)
        for name, f in fits.items():
            pred = np.asarray(f.predict(ex), float).ravel()
            rows.append(
                {
                    "scene": sc.name,
                    "seed": seed,
                    "arm": name,
                    "n_train": int(wmask.sum()),
                    "window_rmse_db": rmse(et, pred),
                    "seconds": f.seconds,
                    "final_pde_loss": f.extra.get("final_pde_loss"),
                    "final_data_loss": f.extra.get("final_data_loss"),
                }
            )
            if seed == seeds[0]:
                keep[name] = pred
        print(f"   [{sc.name}] helmholtz window seed {seed} done", flush=True)
    keep["truth"] = et
    keep["x"] = ex
    keep["train_x"] = x
    return pd.DataFrame(rows), keep


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

ARM_STYLE = {
    "oracle": ("#8a8985", "--"),
    "physics": ("#2a78d6", "-"),
    "kriging": ("#2a78d6", ":"),
    "pinn": ("#1baf7a", "-"),
    "pinn_free": ("#1baf7a", "--"),
    "nn": ("#4a3aa7", "-"),
    "gp": ("#eb6834", "-"),
    "pinn_helmholtz": ("#1baf7a", ":"),
}
ARM_NAME = {
    "oracle": "oracle (free-space law, true tx)",
    "physics": "physics (law fitted)",
    "kriging": "kriging (law + GP)",
    "pinn": "PINN (law + NN)",
    "pinn_free": "PINN, w=1",
    "nn": "black-box NN",
    "gp": "GP (ordinary kriging)",
    "pinn_helmholtz": "Helmholtz PINN",
}


def _walls(ax, scene: S.Scene, color="#ffffff"):
    import matplotlib.patches as mp

    for b in scene.blocks:
        ax.add_patch(
            mp.Rectangle(
                (b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fc="none", ec=color, lw=0.8
            )
        )


def _crop(sc: Scenario, arr: np.ndarray):
    f = sc.field
    i = (f.x >= 0) & (f.x <= sc.scene.width)
    j = (f.y >= 0) & (f.y <= sc.scene.height)
    return arr[np.ix_(j, i)], (0.0, sc.scene.width, 0.0, sc.scene.height)


def fig_scene(scs: dict[str, Scenario]):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    w = scs["walls"]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.2))
    panels = [
        (
            axes[0, 0],
            w,
            w.field.power_db(),
            "walls: raw power |u|$^2$ [dB], with fading",
        ),
        (axes[0, 1], w, w.local_mean, "walls: local-mean power [dB] (the target)"),
        (
            axes[1, 0],
            scs["free"],
            scs["free"].local_mean,
            "free space: local-mean power [dB]",
        ),
    ]
    vmax = float(np.nanmax(w.local_mean[w.field.interior_mask()]))
    for ax, sc, arr, title in panels:
        a, ext = _crop(sc, arr)
        im = ax.imshow(
            a, origin="lower", extent=ext, cmap="magma", vmin=vmax - 45, vmax=vmax
        )
        _walls(ax, w.scene)
        ax.plot(*sc.scene.tx, marker="*", color="#ffffff", ms=11, mec="#0b0b0b")
        ax.grid(False)
        P._style(ax, title, "x [$\\lambda$]", "y [$\\lambda$]")
        fig.colorbar(im, ax=ax, shrink=0.85)
    ax = axes[1, 1]
    d = w.local_mean - scs["free"].local_mean
    a, ext = _crop(w, d)
    im = ax.imshow(a, origin="lower", extent=ext, cmap="RdBu_r", vmin=-25, vmax=25)
    _walls(ax, w.scene, "#0b0b0b")
    ax.grid(False)
    P._style(
        ax, "what the walls do: walls - free [dB]", "x [$\\lambda$]", "y [$\\lambda$]"
    )
    fig.colorbar(im, ax=ax, shrink=0.85)
    P.save(fig, FIG, "rf_scene")


def fig_convergence(conv: pd.DataFrame, analytic: pd.DataFrame, ppw: int, tol: float):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    c = conv[conv.ppw < conv.ppw.max()]
    ax = axes[0]
    ax.plot(
        c.ppw,
        c.local_mean_p99_db,
        marker="o",
        color=ARM_STYLE["physics"][0],
        label="local mean, 99th percentile",
    )
    ax.plot(
        c.ppw,
        c.local_mean_rms_db,
        marker="s",
        color=ARM_STYLE["physics"][0],
        ls="--",
        label="local mean, rms",
    )
    ax.plot(
        c.ppw,
        c.raw_rms_db,
        marker="^",
        color=ARM_STYLE["nn"][0],
        label="raw |u|$^2$ with fading, rms",
    )
    ax.axhline(tol, color=P.INK_MUTED, ls=":", lw=1.2, label=f"tolerance {tol} dB")
    ax.axvline(ppw, color=P.INK_MUTED, lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, labelcolor=P.INK_2)
    P._style(
        ax,
        f"walls: change against {int(conv.ppw.max())} ppw",
        "points per wavelength",
        "|difference|  [dB]",
    )
    ax = axes[1]
    ax.plot(
        analytic.ppw,
        analytic.rms_db_error,
        marker="o",
        color=ARM_STYLE["physics"][0],
        label="rms",
    )
    ax.plot(
        analytic.ppw,
        analytic.max_abs_db_error,
        marker="s",
        color=ARM_STYLE["gp"][0],
        label="max",
    )
    g = analytic.ppw.to_numpy(float)
    ax.plot(
        g,
        analytic.rms_db_error.iloc[0] * (g[0] / g) ** 2,
        color=P.INK_MUTED,
        ls=":",
        label="$h^2$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=8, labelcolor=P.INK_2)
    P._style(
        ax,
        "free space: solver against (i/4) H$_0^{(1)}$(kr)",
        "points per wavelength",
        "power error  [dB]",
    )
    P.save(fig, FIG, "rf_convergence")


def _mapshow(ax, xy: np.ndarray, vals: np.ndarray, **kw):
    """Scattered lattice values as an image; missing lattice points stay blank."""
    i = np.rint((xy[:, 0] - xy[:, 0].min()) / EVAL_STEP).astype(int)
    j = np.rint((xy[:, 1] - xy[:, 1].min()) / EVAL_STEP).astype(int)
    img = np.full((j.max() + 1, i.max() + 1), np.nan)
    img[j, i] = vals
    h = EVAL_STEP / 2
    ext = (
        xy[:, 0].min() - h,
        xy[:, 0].max() + h,
        xy[:, 1].min() - h,
        xy[:, 1].max() + h,
    )
    return ax.imshow(img, origin="lower", extent=ext, interpolation="nearest", **kw)


def fig_reconstruction(sc: Scenario, fits: dict[str, Fit], idx: np.ndarray, tag: str):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    names = [
        a for a in ("physics", "kriging", "pinn", "pinn_free", "nn", "gp") if a in fits
    ]
    ncol = 1 + len(names)
    fig, axes = plt.subplots(2, ncol, figsize=(2.9 * ncol, 5.0))
    ex = sc.eval_x
    vmax = float(np.max(sc.eval_truth))
    kw = dict(cmap="magma", vmin=vmax - 40, vmax=vmax)
    ax = axes[0, 0]
    im = _mapshow(ax, ex, sc.eval_truth, **kw)
    ax.scatter(
        sc.pool_x[idx, 0],
        sc.pool_x[idx, 1],
        s=7,
        c="#ffffff",
        edgecolors="#0b0b0b",
        lw=0.4,
    )
    ax.plot(*sc.scene.tx, marker="*", color="#ffffff", ms=10, mec="#0b0b0b")
    P._style(ax, f"truth + {len(idx)} receivers")
    axes[1, 0].axis("off")
    for k, name in enumerate(names, start=1):
        f = fits[name]
        pred = np.asarray(f.predict(ex), float)
        a = axes[0, k]
        _mapshow(a, ex, pred, **kw)
        if "xt" in f.params:
            a.plot(
                f.params["xt"], f.params["yt"], marker="x", color="#1baf7a", ms=9, mew=2
            )
        a.plot(*sc.scene.tx, marker="*", color="#ffffff", ms=8, mec="#0b0b0b")
        P._style(a, ARM_NAME[name])
        b = axes[1, k]
        err = pred - sc.eval_truth
        im_err = _mapshow(b, ex, err, cmap="RdBu_r", vmin=-15, vmax=15)
        P._style(b, f"error, rms {rmse(sc.eval_truth, pred):.1f} dB")
    for a in axes.ravel():
        if a.axison:
            if sc.name == "walls":
                _walls(a, sc.scene, "#8a8985")
            a.set_xlim(0, sc.scene.width)
            a.set_ylim(0, sc.scene.height)
            a.set_aspect("equal")
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])
    fig.colorbar(im, ax=axes[0, -1], shrink=0.9, label="power [dB]")
    fig.colorbar(im_err, ax=axes[1, -1], shrink=0.9, label="prediction - truth [dB]")
    P.save(fig, FIG, f"rf_reconstruction_{tag}")


def fig_tx(budget: pd.DataFrame, scenes: dict[str, S.Scene]):
    """Where the fits put the transmitter, zoomed on it, and the median miss."""
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0))
    budgets = sorted(budget.n_train.unique())
    cmap = plt.get_cmap("viridis")
    for ax, name in zip(axes[:2], ("free", "walls")):
        scene = scenes[name]
        d = budget[(budget.scene == name) & (budget.arm == "physics")]
        _walls(ax, S.floor_plan(), P.INK_MUTED if name == "walls" else P.GRID)
        for k, nb in enumerate(budgets):
            e = d[d.n_train == nb]
            ax.scatter(
                e.param_xt,
                e.param_yt,
                s=28,
                color=cmap(k / max(len(budgets) - 1, 1)),
                edgecolors=P.SURFACE,
                lw=0.5,
                label=f"{nb} receivers",
                zorder=3,
            )
        ax.plot(*scene.tx, marker="*", color="#0b0b0b", ms=14, ls="none", zorder=4)
        ax.set_xlim(scene.tx[0] - 4.5, scene.tx[0] + 5.5)
        ax.set_ylim(scene.tx[1] - 4.0, scene.tx[1] + 4.0)
        ax.set_aspect("equal")
        P._style(
            ax,
            f"{name}: physics fits (star = truth)",
            "x [$\\lambda$]",
            "y [$\\lambda$]",
        )
    axes[0].legend(fontsize=8, labelcolor=P.INK_2, loc="lower left")
    ax = axes[2]
    for name, ls in (("free", "--"), ("walls", "-")):
        for arm in ("physics", "pinn_free"):
            e = budget[(budget.scene == name) & (budget.arm == arm)]
            if e.empty:
                continue
            m = e.groupby("n_train").tx_err_lambda.median()
            ax.plot(
                m.index,
                m.values,
                ls=ls,
                marker="o",
                ms=4,
                color=ARM_STYLE[arm][0],
                label=f"{name}, {arm}",
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    _budget_ticks(ax, budgets)
    ax.legend(fontsize=8, labelcolor=P.INK_2)
    P._style(ax, "median distance to the true transmitter", "receivers", "[$\\lambda$]")
    P.save(fig, FIG, "rf_tx_recovery")


def _budget_ticks(ax, budgets) -> None:
    from matplotlib.ticker import NullFormatter

    ax.set_xticks(list(budgets))
    ax.set_xticklabels([str(int(b)) for b in budgets])
    ax.xaxis.set_minor_formatter(NullFormatter())


def fig_budget(budget: pd.DataFrame):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for ax, name in zip(axes, ("free", "walls")):
        d = (
            budget[budget.scene == name]
            .groupby(["arm", "n_train"])
            .map_rmse_db.median()
            .reset_index()
        )
        for arm in ARMS:
            e = d[d.arm == arm]
            c, ls = ARM_STYLE[arm]
            ax.plot(
                e.n_train,
                e.map_rmse_db,
                color=c,
                ls=ls,
                marker="o",
                ms=4,
                label=ARM_NAME[arm],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        _budget_ticks(ax, sorted(d.n_train.unique()))
        P._style(
            ax,
            f"{name}: map error vs number of receivers",
            "receivers",
            "rmse vs true map  [dB]",
        )
    axes[1].legend(fontsize=8, labelcolor=P.INK_2)
    P.save(fig, FIG, "rf_budget")


def fig_extrapolation(ex: pd.DataFrame):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, name in zip(axes, ("free", "walls")):
        d = (
            ex[ex.scene == name]
            .groupby("arm")[["map_rmse_room_db", "map_rmse_beyond_db"]]
            .median()
        )
        arms = [a for a in ARMS if a in d.index]
        xs = np.arange(len(arms))
        for i, a in enumerate(arms):
            c = ARM_STYLE[a][0]
            ax.bar(xs[i] - 0.2, d.loc[a, "map_rmse_room_db"], 0.38, color=c, alpha=0.45)
            ax.bar(xs[i] + 0.2, d.loc[a, "map_rmse_beyond_db"], 0.38, color=c)
        ax.set_xticks(xs)
        ax.set_xticklabels(arms, rotation=30, fontsize=8)
        ax.set_yscale("log")
        P._style(
            ax,
            f"{name}: trained in the transmitter's room",
            None,
            "rmse vs true map [dB]",
        )
    axes[0].bar(np.nan, np.nan, color=P.INK_MUTED, alpha=0.45, label="the room (x < 9)")
    axes[0].bar(np.nan, np.nan, color=P.INK_MUTED, label="beyond (x > 10)")
    axes[0].legend(fontsize=8, labelcolor=P.INK_2)
    P.save(fig, FIG, "rf_extrapolation")


def fig_window(keep: dict[str, dict], scs: dict[str, Scenario]):
    import matplotlib.pyplot as plt

    from physprior.viz import plots as P

    P.use_style()
    names = ["truth", "physics", "nn", "gp", "pinn_helmholtz"]
    fig, axes = plt.subplots(
        len(keep), len(names), figsize=(2.6 * len(names), 2.8 * len(keep))
    )
    axes = np.atleast_2d(axes)
    for r, (scene, k) in enumerate(keep.items()):
        vmax = float(np.max(k["truth"]))
        for c, n in enumerate(names):
            a = axes[r, c]
            _mapshow(a, k["x"], k[n], cmap="magma", vmin=vmax - 25, vmax=vmax)
            if n == "truth":
                a.scatter(
                    k["train_x"][:, 0],
                    k["train_x"][:, 1],
                    s=5,
                    c="#ffffff",
                    edgecolors="#0b0b0b",
                    lw=0.3,
                )
                title = f"{scene}: truth"
            else:
                title = f"{ARM_NAME[n]}, {rmse(k['truth'], k[n]):.1f} dB"
            if scene == "walls":
                _walls(a, scs["walls"].scene, "#8a8985")
            x0, x1, y0, y1 = WINDOW
            a.set_xlim(x0, x1)
            a.set_ylim(y0, y1)
            a.set_aspect("equal")
            a.grid(False)
            a.set_xticks([])
            a.set_yticks([])
            P._style(a, title)
    P.save(fig, FIG, "rf_helmholtz_window")


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------


def _meta_base(quick: bool) -> dict[str, Any]:
    return {
        "track": TRACK,
        "data": "simulated (2-D Helmholtz); no real dataset is used",
        "frequency_ghz": S.FREQ_GHZ,
        "wavelength_m": S.WAVELENGTH_M,
        "eps_concrete": [S.EPS_CONCRETE.real, S.EPS_CONCRETE.imag],
        "scene_lambda": [S.WIDTH, S.HEIGHT],
        "tx_true": list(S.TX_TRUE),
        "noise_db": NOISE_DB,
        "n_pool": N_POOL,
        "avg_radius_lambda": S.AVG_RADIUS,
        "p0_free_space_db": S.P0_FREE_SPACE,
        "n_free_space": S.N_FREE_SPACE,
        "window": list(WINDOW),
        "quick": quick,
    }


def _threads() -> None:
    import os
    import warnings

    import torch
    from sklearn.exceptions import ConvergenceWarning

    torch.set_num_threads(int(os.environ.get("PHYSPRIOR_TORCH_THREADS", "2")))
    # A GP hyperparameter at its bound is recorded per fit (`length_at_bound`);
    # the same fact printed hundreds of times is noise.
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


PPW_TOL_DB = 0.5


def solver_study(quick: bool = False) -> dict:
    """Analytic check and grid convergence (rule 4); chooses the resolution."""
    print(f"[{TRACK}] solver checks", flush=True)
    if quick:
        analytic = pd.DataFrame([S.analytic_check(p) for p in (8, 12)])
        conv = pd.DataFrame(S.grid_convergence("walls", ppws=(6, 8, 12)))
        ppw = PPW_QUICK
    else:
        analytic = pd.DataFrame([S.analytic_check(p) for p in (8, 12, 16, 24)])
        conv = pd.DataFrame(S.grid_convergence("walls", ppws=S.CONVERGENCE_PPWS))
        ppw = S.choose_ppw(conv.to_dict("records"), PPW_TOL_DB)
    save_table(analytic, TRACK, "analytic_check")
    save_table(conv, TRACK, "grid_convergence")
    out = {
        "ppw": ppw,
        "ppw_tolerance_db": PPW_TOL_DB,
        "ppw_rule": "coarsest grid within tolerance of the finest at 99% of points",
        "quick": quick,
    }
    save_json(out, TRACK, "solver")
    fig_convergence(conv, analytic, ppw, PPW_TOL_DB)
    print(f"   chose {ppw} points per wavelength", flush=True)
    return out


def scenarios(ppw: int) -> dict[str, Scenario]:
    """Both scenes at `ppw`, with their truth maps and receivers saved."""
    scs = {n: build_scenario(n, ppw) for n in ("free", "walls")}
    oracle = {}
    for n, sc in scs.items():
        save_table(
            pd.DataFrame(
                {"x": sc.eval_x[:, 0], "y": sc.eval_x[:, 1], "truth_db": sc.eval_truth}
            ),
            TRACK,
            f"truth_map_{n}",
        )
        save_table(
            pd.DataFrame(
                {
                    "x": sc.pool_x[:, 0],
                    "y": sc.pool_x[:, 1],
                    "truth_db": sc.pool_truth,
                    "reading_db": sc.pool_noisy,
                }
            ),
            TRACK,
            f"receivers_{n}",
        )
        # How far the oracle's law is from each true map: a property of the
        # scene, not of any arm.
        o = law_np(sc.eval_x, S.P0_FREE_SPACE, S.N_FREE_SPACE, *sc.scene.tx)
        oracle[n] = {
            "oracle_map_rmse_db": rmse(sc.eval_truth, o),
            "solve_seconds": sc.field.seconds,
            "n_eval": len(sc.eval_truth),
        }
    save_json(oracle, TRACK, "scenes")
    fig_scene(scs)
    return scs


# The cached tuning result. Renamed from "nn_cfg" when tuning moved from the
# whole pool to the transmitter's room, so the old choice is not reused.
NN_CFG_NAME = "nn_cfg_room"


def nn_configs(scs: dict[str, Scenario], quick: bool) -> dict[str, dict]:
    """Tuned once per scene, on the tuning seeds, and cached in results/."""
    from physprior.io import load_json

    if quick:
        base = {"width": 32, "depth": 3, "weight_decay": 1e-4}
        return {n: dict(base) for n in scs}
    try:
        return load_json(TRACK, NN_CFG_NAME)
    except FileNotFoundError:
        pass
    cfgs = {n: tune_nn(sc) for n, sc in scs.items()}
    save_json(cfgs, TRACK, NN_CFG_NAME)
    return cfgs


def _clean(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "tuned_val_rmse"}


def sweeps_study(scs, nn_cfgs, quick: bool = False) -> dict:
    epochs = EPOCHS if not quick else 600
    probs = {
        n: make_problem(sc, _clean(nn_cfgs[n]) | {"epochs": epochs})
        for n, sc in scs.items()
    }
    for p in probs.values():
        p.pinn_epochs = epochs

    budgets = [16, 32, 64, 128, 256] if not quick else [32]
    noise = [0.0, 2.0, 4.0, 8.0] if not quick else [2.0]
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    arms = ARMS if not quick else ("oracle", "physics", "pinn", "nn", "gp", "kriging")
    b_rows, n_rows, e_rows = [], [], []
    for n in ("free", "walls"):
        b_rows.append(sweep_budget(probs[n], scs[n], budgets, arms, seeds))
        n_rows.append(sweep_noise(probs[n], scs[n], noise, 128, arms, seeds))
        e_rows.append(study_extrapolation(probs[n], scs[n], arms, seeds))
    budget = pd.concat(b_rows, ignore_index=True)
    save_table(budget, TRACK, "sweep_budget")
    save_table(pd.concat(n_rows, ignore_index=True), TRACK, "sweep_noise")
    ex = pd.concat(e_rows, ignore_index=True)
    save_table(ex, TRACK, "extrapolation")

    # headline reconstructions for the figures, on the first reporting seed
    nb = 128 if not quick else 32
    head = {}
    for n in ("free", "walls"):
        itr, _ = split_random(len(probs[n]), nb, REPORT_SEEDS[0])
        fits = {
            a: fit_arm(a, probs[n], itr, REPORT_SEEDS[0]) for a in arms if a != "oracle"
        }
        fig_reconstruction(scs[n], fits, itr, n)
        head[n] = {
            a: {"params": f.params, "sigma": f.param_sigma, **map_scores(f, scs[n])}
            for a, f in fits.items()
        }
    save_json(head, TRACK, "headline")
    fig_tx(budget, {n: sc.scene for n, sc in scs.items()})
    fig_budget(budget)
    fig_extrapolation(ex)
    return head


def window_study(scs, nn_cfgs, quick: bool = False) -> pd.DataFrame:
    """Shape A on the window, both scenes."""
    seeds = REPORT_SEEDS if not quick else REPORT_SEEDS[:1]
    w_rows, keep = [], {}
    for n in ("free", "walls"):
        df, k = helmholtz_window_study(
            scs[n],
            _clean(nn_cfgs[n]),
            seeds=seeds,
            epochs=HELMHOLTZ_EPOCHS if not quick else 50,
        )
        w_rows.append(df)
        keep[n] = k
    out = pd.concat(w_rows, ignore_index=True)
    save_table(out, TRACK, "helmholtz_window")
    fig_window(keep, scs)
    return out


def figures_from_results() -> None:
    """Redraw the figures that need only the saved tables."""
    from physprior.io import load_table

    budget = load_table(TRACK, "sweep_budget")
    fig_tx(budget, {n: S.SCENES[n]() for n in ("free", "walls")})
    fig_budget(budget)
    fig_extrapolation(load_table(TRACK, "extrapolation"))


def run_stage(stage: str, quick: bool = False) -> None:
    """One stage on its own, reusing the solver study's resolution.

    Lets the sweeps and the (slow) shape-A window run in parallel processes
    after `solver_study` has run once.
    """
    from physprior.io import load_json

    _threads()
    ppw = int(load_json(TRACK, "solver")["ppw"])
    scs = (
        scenarios(ppw)
        if stage == "sweeps"
        else {n: build_scenario(n, ppw) for n in ("free", "walls")}
    )
    cfgs = nn_configs(scs, quick)
    {"sweeps": sweeps_study, "window": window_study}[stage](scs, cfgs, quick)


def run(quick: bool = False) -> dict:
    """The whole track, in order. Writes results/fields/rf/meta.json."""
    _threads()
    t0 = time.time()
    solver = solver_study(quick)
    scs = scenarios(solver["ppw"])
    cfgs = nn_configs(scs, quick)
    sweeps_study(scs, cfgs, quick)
    window_study(scs, cfgs, quick)
    meta = finalise(quick, seconds=time.time() - t0)
    print(f"[{TRACK}] done in {meta['seconds'] / 60:.1f} min", flush=True)
    return meta


def finalise(quick: bool = False, seconds: float | None = None) -> dict:
    """Gather the stages' outputs into meta.json and, for a full run, the docs."""
    from physprior.io import load_json

    meta = _meta_base(quick)
    for name, file in (
        ("solver", "solver"),
        ("scenes", "scenes"),
        ("nn_cfg", NN_CFG_NAME),
        ("headline", "headline"),
    ):
        try:
            meta[name] = load_json(TRACK, file)
        except FileNotFoundError:
            meta[name] = None
    meta["seconds"] = seconds
    save_json(meta, TRACK, "meta")
    if not quick:
        # a quick run's numbers are not the study's; they must not reach docs/
        render_doc()
    return meta


# ---------------------------------------------------------------------------
# the docs page, generated from results/ only
# ---------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, fmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float | np.floating):
                cells.append("–" if not np.isfinite(v) else fmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def summary_tables() -> dict[str, pd.DataFrame]:
    """The medians the docs page and the notebook show."""
    from physprior.io import load_table

    budget = load_table(TRACK, "sweep_budget")
    ex = load_table(TRACK, "extrapolation")
    noise = load_table(TRACK, "sweep_noise")
    out = {}
    out["budget"] = (
        budget.pivot_table(
            index=["scene", "arm"],
            columns="n_train",
            values="map_rmse_db",
            aggfunc="median",
        )
        .reset_index()
        .rename(columns=lambda c: f"n={c}" if isinstance(c, int | np.integer) else c)
    )
    out["tx"] = (
        budget[budget.arm.isin(["physics", "pinn", "pinn_free"])]
        .groupby(["scene", "arm", "n_train"])[["tx_err_lambda", "param_n", "param_P0"]]
        .median()
        .reset_index()
    )
    out["extrapolation"] = (
        ex.groupby(["scene", "arm"])[
            ["map_rmse_room_db", "map_rmse_beyond_db", "tx_err_lambda"]
        ]
        .median()
        .reset_index()
    )
    out["noise"] = (
        noise.pivot_table(
            index=["scene", "arm"],
            columns="noise_db",
            values="map_rmse_db",
            aggfunc="median",
        )
        .reset_index()
        .rename(columns=lambda c: f"{c:g} dB" if isinstance(c, float) else c)
    )
    try:
        w = load_table(TRACK, "helmholtz_window")
        out["window"] = (
            w.groupby(["scene", "arm"])[["window_rmse_db", "seconds"]]
            .median()
            .reset_index()
        )
    except FileNotFoundError:
        pass
    return out


def _richardson_note(conv: pd.DataFrame, ppw: int) -> str:
    """The difference against the finest grid understates the error of the
    chosen grid: for a second-order scheme the true error is larger by
    1 / (1 - (h_ref / h)^2)."""
    ref = int(conv.ppw.max())
    row = conv[conv.ppw == ppw]
    if ppw == ref or row.empty:
        return "The finest grid is used; its own error is not estimated."
    f = 1.0 / (1.0 - (ppw / ref) ** 2)
    r = row.iloc[0]
    return (
        f"Assuming second order (the analytic check above falls as h²), the error of "
        f"the {ppw}-ppw map against the continuum is about {f:.1f}× its difference "
        f"from {ref} ppw: {f * r.local_mean_rms_db:.2f} dB rms and "
        f"{f * r.local_mean_p99_db:.2f} dB at the 99th percentile, against "
        f"{NOISE_DB:g} dB of receiver noise."
    )


def render_doc(path=None) -> str:
    """Write docs/fields/rf.md (or `path`) from results/fields/rf."""
    from physprior.io import load_json, load_table

    meta = load_json(TRACK, "meta")
    conv = load_table(TRACK, "grid_convergence")
    analytic = load_table(TRACK, "analytic_check")
    t = summary_tables()
    fig = "../../figures/fields"
    conv_show = conv[
        [
            "ppw",
            "n_unknowns",
            "seconds",
            "local_mean_p99_db",
            "local_mean_max_db",
            "local_mean_rms_db",
            "raw_rms_db",
        ]
    ]
    parts = [
        "# fields / rf: radio propagation",
        "",
        "Where is the transmitter, and what is the received-power map, given a few",
        "noisy readings? A simulated track: the field is solved from the 2-D",
        "Helmholtz equation, so the true map and the true transmitter position are",
        "known and every arm is scored against them.",
        "",
        "Code: [`src/physprior/problems/fields/rf.py`](../../src/physprior/problems/fields/rf.py),",
        "[`rf_sim.py`](../../src/physprior/problems/fields/rf_sim.py)",
        "· Results: [`results/fields/rf`](../../results/fields/rf)",
        "",
        "This page is generated by `physprior.problems.fields.rf.render_doc()`.",
        "",
        "## The simulation",
        "",
        f"A {meta['frequency_ghz']:g} GHz transmitter ({meta['wavelength_m'] * 100:.1f} cm wavelength)",
        f"in a {meta['scene_lambda'][0]:g} × {meta['scene_lambda'][1]:g} wavelength floor plan. The",
        "field solves ∇²u + k²n(x)²u = −f with second-order finite differences, a",
        "perfectly matched layer, and a sparse LU solve. Walls are concrete,",
        f"ε = {meta['eps_concrete'][0]:.2f} + {meta['eps_concrete'][1]:.2f}i (ITU-R P.2040-1).",
        f"A receiver reads the mean of |u|² over a disk of radius {meta['avg_radius_lambda']:g} λ,",
        f"in dB, with {meta['noise_db']:g} dB Gaussian noise. In two dimensions a source spreads",
        "cylindrically, so the free-space path-loss exponent is n = 1, not 2.",
        "",
        "![scene](" + fig + "/rf_scene.png)",
        "",
        "### Convergence",
        "",
        "Against the analytic free-space field (i/4)H₀⁽¹⁾(kr), power error in dB:",
        "",
        _md_table(
            analytic[["ppw", "rms_db_error", "max_abs_db_error", "seconds"]], "{:.3g}"
        ),
        "",
        "Walls scene, change against the finest grid, in dB:",
        "",
        _md_table(conv_show, "{:.3g}"),
        "",
        f"Resolution used: **{meta['solver']['ppw']} points per wavelength**, the coarsest grid whose",
        f"local-mean map is within {meta['solver']['ppw_tolerance_db']:g} dB of the finest at 99% of points. The raw",
        "power with fading converges more slowly, because the scheme's phase error",
        "accumulates over the scene; the arms are not asked to reproduce it.",
        "",
        _richardson_note(conv, int(meta["solver"]["ppw"])),
        "",
        "![convergence](" + fig + "/rf_convergence.png)",
        "",
        "## The arms",
        "",
        "`physics` fits P = P0 − 10 n log10(d), d the distance to an unknown",
        "transmitter (x_t, y_t), from several starts. `oracle` is the same law with",
        "the true position, n = 1 and the analytic P0. `pinn` is the law plus a",
        "network correction, started from the physics fit, with a physics weight",
        "tuned per scene on the tuning seeds (pinned at the top of its grid in both",
        "scenes, so the arm is the law by choice); `pinn_free` is the same with",
        "w_phys = 1.",
        "`nn` is a tuned MLP, `gp` is ordinary kriging, `kriging` is the fitted law",
        "plus a GP on its residuals.",
        "",
        f"Oracle map error: free space {meta['scenes']['free']['oracle_map_rmse_db']:.2f} dB, walls "
        f"{meta['scenes']['walls']['oracle_map_rmse_db']:.2f} dB.",
        "",
        "## Map error against the number of receivers",
        "",
        "Median over reporting seeds of the RMSE against the true map, dB:",
        "",
        _md_table(t["budget"]),
        "",
        "![budget](" + fig + "/rf_budget.png)",
        "",
        "## Where is the transmitter?",
        "",
        "Median distance from the true position (wavelengths), and the fitted",
        "exponent and P0:",
        "",
        _md_table(t["tx"]),
        "",
        "![tx](" + fig + "/rf_tx_recovery.png)",
        "",
        "## Reconstructions (128 receivers, seed 11)",
        "",
        "![free](" + fig + "/rf_reconstruction_free.png)",
        "",
        "![walls](" + fig + "/rf_reconstruction_walls.png)",
        "",
        "## Extrapolation: trained in the transmitter's room",
        "",
        _md_table(t["extrapolation"]),
        "",
        "![extrapolation](" + fig + "/rf_extrapolation.png)",
        "",
        "## Noise (128 receivers, scored against the clean map)",
        "",
        _md_table(t["noise"]),
        "",
    ]
    if "window" in t:
        parts += [
            "## Shape A: a Helmholtz-residual PINN on a window",
            "",
            f"Window x, y ∈ {meta['window']} λ, all pool receivers in it. The PINN is",
            "given the true refractive-index map; the readings carry no phase. w_pde = 1,",
            "untuned.",
            "",
            _md_table(t["window"]),
            "",
            "![window](" + fig + "/rf_helmholtz_window.png)",
            "",
        ]
    path = Path(path) if path else get_settings().root / "docs" / "fields" / "rf.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n")
    return str(path)
