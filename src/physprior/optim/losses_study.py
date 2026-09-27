"""Loss functions, for a physics model and a black box, under three kinds of noise.

The loss is the likelihood a fit assumes. MSE is the Gaussian likelihood;
L1 the Laplace; Huber is Gaussian in the core and Laplace in the tails;
Cauchy (Lorentzian) is the likelihood of a noise so heavy-tailed it has no
variance; the Gaussian NLL with a learned sigma lets the fit also estimate
how noisy each region is (heteroscedastic).

For a physics model the question is bias in the recovered constant: an
estimator whose likelihood is wrong for the noise can pull the constant off
its true value. For the black box the question is prediction: a loss that
chases outliers bends the curve toward them.

Data: the damped oscillator of `problems.oscillator` (gamma and omega0
unknown), with

    gaussian    N(0, 0.05)
    student_t   Student-t, nu = 2 (infinite variance), scale 0.05
    outliers    N(0, 0.05) plus 10% of points displaced by 0.5-1.0

Robust scales. Huber's delta and Cauchy's c are set from the noise scale,
not tuned: s = 1.4826 * MAD of the residuals of an MSE fit, delta = 1.345 s
and c = 2.385 s, the standard constants that give 95% efficiency at the
Gaussian (Holland & Welsch 1977). Fits start near the truth (10% log
scatter) so that what is measured is the estimator's bias, not an
optimizer's failure to find the basin; `optimizers_study` measures that.

The second half is the physics weight as a loss hyperparameter: the
residual PINN on the same problem at w_phys from 0 to 100.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from physprior.benchmark.metrics import nrmse
from physprior.methods.base import REPORT_SEEDS
from physprior.methods.neural import DTYPE, mlp
from physprior.optim.problems import (
    OSC_T_OUT,
    OSC_T_TRAIN,
    OSC_TRUTH,
    Model,
    oscillator,
    oscillator_solution,
)

AREA = "optim"
LOSSES = ("mse", "l1", "huber", "cauchy", "gauss_nll")
NOISES = ("gaussian", "student_t", "outliers")
NOISE_SCALE = 0.05
OUTLIER_FRAC = 0.10
N_TRAIN = 40
REALISATIONS = 4  # noise draws per reporting seed for the physics model
W_GRID = (0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
HUBER_K, CAUCHY_K, MAD_TO_SD = 1.345, 2.385, 1.4826


def make_data(noise: str, rng: np.random.Generator):
    t = np.sort(rng.uniform(*OSC_T_TRAIN, N_TRAIN))
    clean = oscillator_solution(t, **OSC_TRUTH)
    if noise == "gaussian":
        eps = rng.normal(0.0, NOISE_SCALE, N_TRAIN)
    elif noise == "student_t":
        eps = NOISE_SCALE * rng.standard_t(2.0, N_TRAIN)
    elif noise == "outliers":
        eps = rng.normal(0.0, NOISE_SCALE, N_TRAIN)
        k = rng.choice(N_TRAIN, round(OUTLIER_FRAC * N_TRAIN), replace=False)
        eps[k] += rng.choice([-1.0, 1.0], len(k)) * rng.uniform(0.5, 1.0, len(k))
    else:
        raise ValueError(f"unknown noise {noise!r}")
    return t, clean + eps


def eval_grids():
    t_in = np.linspace(*OSC_T_TRAIN, 121)
    t_out = np.linspace(OSC_T_OUT[0], OSC_T_OUT[1], 81)[1:]
    t_all = np.concatenate([t_in, t_out])
    scale = float(np.std(oscillator_solution(t_all, **OSC_TRUTH)))
    return t_in, t_out, scale


def robust_loss(kind: str, r: torch.Tensor, scale: float) -> torch.Tensor:
    """Mean loss of residuals r. `scale` is the robust noise scale s."""
    if kind == "mse":
        return torch.mean(r**2)
    if kind == "l1":
        return torch.mean(torch.abs(r))
    if kind == "huber":
        d = HUBER_K * scale
        a = torch.abs(r)
        return torch.mean(torch.where(a <= d, 0.5 * r**2, d * (a - 0.5 * d)))
    if kind == "cauchy":
        c = CAUCHY_K * scale
        return torch.mean(0.5 * c**2 * torch.log1p((r / c) ** 2))
    raise ValueError(f"unknown loss {kind!r}")


def _mad_scale(r: np.ndarray) -> float:
    return float(MAD_TO_SD * np.median(np.abs(r - np.median(r)))) or 1e-3


# ---------------------------------------------------------------------------
# the two models
# ---------------------------------------------------------------------------


def _train(params, loss_fn, steps: int, lr: float) -> None:
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        sched.step()


def fit_physics(t, y, loss: str, rng, scale: float | None = None, steps: int = 1500):
    """Returns (theta dict, predict fn, residual scale used)."""
    tt = torch.tensor(t, dtype=DTYPE).reshape(-1, 1)
    yt = torch.tensor(y, dtype=DTYPE)
    raw = {
        k: torch.tensor(
            np.log(v) + rng.normal(0.0, 0.1), dtype=DTYPE, requires_grad=True
        )
        for k, v in OSC_TRUTH.items()
    }
    # heteroscedastic noise model: log sigma(t) = a + b t
    a = torch.tensor(np.log(np.std(y) * 0.2), dtype=DTYPE, requires_grad=True)
    b = torch.zeros((), dtype=DTYPE, requires_grad=True)

    from physprior.optim.problems import _osc_law_t

    def pred(x):
        return _osc_law_t(x, torch.exp(raw["gamma"]), torch.exp(raw["omega0"]))

    def loss_fn():
        r = pred(tt) - yt
        if loss == "gauss_nll":
            log_s = a + b * tt[:, 0]
            return torch.mean(0.5 * (r * torch.exp(-log_s)) ** 2 + log_s)
        return robust_loss(loss, r, scale or 1.0)

    params = list(raw.values()) + ([a, b] if loss == "gauss_nll" else [])
    _train(params, loss_fn, steps, lr=0.02)
    theta = {k: float(torch.exp(v).detach()) for k, v in raw.items()}

    def predict(tq):
        with torch.no_grad():
            return pred(torch.tensor(tq, dtype=DTYPE).reshape(-1, 1)).numpy()

    return theta, predict


def fit_nn(t, y, loss: str, seed: int, scale: float | None = None, steps: int = 3000):
    """The black box: tanh MLP, width 32 depth 3, the repo's default arm.
    For the Gaussian NLL a second output is log sigma(t)."""
    torch.manual_seed(seed)
    mu_t, sd_t = float(np.mean(t)), float(np.std(t))
    mu_y, sd_y = float(np.mean(y)), float(np.std(y))
    xs = torch.tensor((t - mu_t) / sd_t, dtype=DTYPE).reshape(-1, 1)
    ys = torch.tensor((y - mu_y) / sd_y, dtype=DTYPE)
    net = mlp(1, 32, 3, d_out=2 if loss == "gauss_nll" else 1)
    s_std = (scale or sd_y) / sd_y

    def loss_fn():
        out = net(xs)
        r = out[:, 0] - ys
        if loss == "gauss_nll":
            log_s = out[:, 1]
            return torch.mean(0.5 * (r * torch.exp(-log_s)) ** 2 + log_s)
        return robust_loss(loss, r, s_std)

    opt = torch.optim.Adam(net.parameters(), lr=5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for _ in range(steps):
        opt.zero_grad()
        loss_fn().backward()
        opt.step()
        sched.step()

    def predict(tq):
        with torch.no_grad():
            z = net(torch.tensor((tq - mu_t) / sd_t, dtype=DTYPE).reshape(-1, 1))[:, 0]
        return z.numpy() * sd_y + mu_y

    return predict


# ---------------------------------------------------------------------------
# the loss-function study
# ---------------------------------------------------------------------------


def _cell(args) -> list[dict]:
    torch.set_num_threads(1)
    model, noise, seed, k, steps = args
    rng = np.random.default_rng(10_000 * seed + 100 * k + NOISES.index(noise))
    t, y = make_data(noise, rng)
    t_in, t_out, scale = eval_grids()
    y_in = oscillator_solution(t_in, **OSC_TRUTH)
    y_out = oscillator_solution(t_out, **OSC_TRUTH)
    rows = []
    # The robust scale comes from an MSE fit of the same model.
    if model == "physics":
        _, p0 = fit_physics(t, y, "mse", np.random.default_rng(seed + k), steps=steps)
    else:
        p0 = fit_nn(t, y, "mse", seed, steps=steps)
    s = _mad_scale(y - p0(t))
    for loss in LOSSES:
        t0 = time.time()
        if model == "physics":
            theta, pred = fit_physics(
                t, y, loss, np.random.default_rng(seed + k), scale=s, steps=steps
            )
        else:
            theta, pred = {}, fit_nn(t, y, loss, seed, scale=s, steps=steps)
        row = {
            "model": model,
            "noise": noise,
            "loss": loss,
            "seed": seed,
            "realisation": k,
            "robust_scale": s,
            "nrmse_in": nrmse(y_in, pred(t_in), scale=scale),
            "nrmse_out": nrmse(y_out, pred(t_out), scale=scale),
            "seconds": time.time() - t0,
        }
        for name, v in theta.items():
            row[f"theta_{name}"] = v
            row[f"err_{name}_pct"] = (v - OSC_TRUTH[name]) / OSC_TRUTH[name] * 100.0
        rows.append(row)
    return rows


def run_losses(quick: bool = False, workers: int = 3) -> pd.DataFrame:
    seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS
    reals = 1 if quick else REALISATIONS
    steps_p, steps_n = (200, 300) if quick else (1500, 3000)
    cells = [
        ("physics", nz, s, k, steps_p)
        for nz in NOISES
        for s in seeds
        for k in range(reals)
    ]
    cells += [("nn", nz, s, 0, steps_n) for nz in NOISES for s in seeds]
    if workers <= 1:
        res = [_cell(c) for c in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(_cell, cells))
    return pd.DataFrame([r for rows in res for r in rows])


def summarise_losses(df: pd.DataFrame) -> pd.DataFrame:
    """Bias is the mean signed error of gamma over all fits; spread its
    standard deviation. Prediction errors are medians."""
    g = df.groupby(["model", "noise", "loss"])
    out = g.agg(
        n=("nrmse_in", "size"),
        nrmse_in=("nrmse_in", "median"),
        nrmse_out=("nrmse_out", "median"),
    ).reset_index()
    if "err_gamma_pct" in df:
        b = g["err_gamma_pct"].agg(["mean", "std"]).reset_index()
        b.columns = ["model", "noise", "loss", "gamma_bias_pct", "gamma_sd_pct"]
        out = out.merge(b, on=["model", "noise", "loss"])
        out["gamma_bias_z"] = out["gamma_bias_pct"] / (
            out["gamma_sd_pct"] / np.sqrt(out["n"])
        )
    return out


# ---------------------------------------------------------------------------
# the physics weight as a loss hyperparameter
# ---------------------------------------------------------------------------


def _w_cell(args) -> dict:
    torch.set_num_threads(1)
    noise, w, seed, steps, lr = args
    task = oscillator()
    rng = np.random.default_rng(20_000 + seed)
    t, y = make_data(noise, rng)
    task.x_train, task.y_train = t.reshape(-1, 1), y
    m = Model(task, "pinn", seed, w_phys=w)
    t0 = time.time()
    _train(m.parameters(), m.loss, steps, lr)
    th = m.theta()
    t_in, t_out, scale = eval_grids()
    with torch.no_grad():
        data = float(m.data_loss())
    return {
        "noise": noise,
        "w_phys": w,
        "seed": seed,
        "data_loss": data,
        "phys_loss": float(m.phys_loss().detach()),
        "nrmse_in": nrmse(
            oscillator_solution(t_in, **OSC_TRUTH),
            m.predict(t_in.reshape(-1, 1)),
            scale=scale,
        ),
        "nrmse_out": nrmse(
            oscillator_solution(t_out, **OSC_TRUTH),
            m.predict(t_out.reshape(-1, 1)),
            scale=scale,
        ),
        "err_gamma_pct": (th["gamma"] - OSC_TRUTH["gamma"]) / OSC_TRUTH["gamma"] * 100,
        "err_omega0_pct": (th["omega0"] - OSC_TRUTH["omega0"])
        / OSC_TRUTH["omega0"]
        * 100,
        "seconds": time.time() - t0,
    }


def pinn_adam_lr(default: float = 1e-2) -> float:
    """The Adam rate the optimizer study chose for the oscillator PINN on the
    TUNING seeds, if it has run; otherwise a fixed default."""
    try:
        from physprior.io import load_table

        c = load_table(AREA, "lr_chosen")
        r = c[(c.task == "oscillator") & (c.model == "pinn") & (c.optimizer == "adam")]
        return float(r.lr.iloc[0]) if len(r) else default
    except (FileNotFoundError, KeyError, IndexError):
        return default


def run_w_dial(quick: bool = False, workers: int = 3) -> pd.DataFrame:
    seeds = REPORT_SEEDS[:1] if quick else REPORT_SEEDS
    weights = (0.0, 1.0) if quick else W_GRID
    steps = 200 if quick else 3000
    lr = pinn_adam_lr()
    cells = [
        (nz, w, s, steps, lr)
        for nz in ("gaussian", "outliers")
        for w in weights
        for s in seeds
    ]
    if workers <= 1:
        rows = [_w_cell(c) for c in cells]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_w_cell, cells))
    out = pd.DataFrame(rows)
    out["adam_lr"] = lr
    return out


def run(quick: bool = False, workers: int | None = None) -> dict[str, pd.DataFrame]:
    from physprior.io import save_table

    workers = workers if workers is not None else (1 if quick else 3)
    t0 = time.time()
    df = run_losses(quick, workers)
    save_table(df, AREA, "losses")
    summ = summarise_losses(df)
    save_table(summ, AREA, "losses_summary")
    print(f"  losses: {len(df)} fits, {time.time() - t0:.0f}s", flush=True)
    t0 = time.time()
    wd = run_w_dial(quick, workers)
    save_table(wd, AREA, "w_phys_dial")
    print(f"  w_phys dial: {len(wd)} fits, {time.time() - t0:.0f}s", flush=True)
    return {"losses": df, "summary": summ, "w_dial": wd}
