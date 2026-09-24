"""The regime where a physics prior should win, and a controlled test of it.

Every real-data track in this package has a law that is either **exact**
(Kepler on a two-body system) or **unidentifiable from the band observed**
(Planck on FIRAS). Neither is where a physics-informed model earns its keep,
and the honest scorecard in `docs/plans/PLAN.md` says so: the `pinn` arm wins
one cell out of twelve.

The case it was built for is the third one, and it was missing: **the law is
right as far as it goes, and something real has been left out of it.** That
is the normal condition of applied physics -- a neglected oblateness term, a
higher post-Newtonian order, a planet's own mass, an unmodelled instrument
response.

## The system

    y(r)  =  GM / r^2            the law we model
           + eps * GM * R / r^3  a term we NEGLECT, and never tell any arm
           + measurement noise

The `1/r^3` shape is not arbitrary: that is how a real neglected term looks,
one order higher in `1/r`, from an oblateness or a first relativistic
correction. `eps` is the dial. At `eps = 0` the law is exact and this reduces
to the case the rest of the package already covers.

## What each arm can do about it

`physics`  fits `GM` inside the law it was given. It CANNOT represent the
           neglected term at all, so as `eps` grows it absorbs what it can
           into `GM` -- the parameter goes wrong, and the fit floors out.
`nn`       has to learn the whole function from scratch, including the part
           the law would have handed it for free. With little data that is
           hopeless, and the failure is worst where the signal is smooth and
           boring.
`pinn`     gets the law AND a correction network, so the network only has to
           represent `eps * GM * R / r^3` -- a small, smooth residual. That is
           a far easier learning problem than the whole curve.

So the prediction is a CROSSOVER: `physics` best at small `eps`, `pinn` best
in the middle, and `nn` closing only when there is enough data that learning
the whole function stops being the hard part. Whether that happens is the
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from physprior.benchmark.metrics import nrmse
from physprior.methods.base import REPORT_SEEDS

# Reference scales, chosen so every quantity is O(1) and the loss surface is
# usable -- the same reason the rest of the package works in AU and years.
GM_TRUE = 1.0
R_REF = 1.0


@dataclass
class NeglectedSystem:
    """The truth, and the part of it we pretend not to know.

    `shape` decides whether a physics prior can help at all, and it is the
    most important knob here:

    `power`  the missing term is `eps GM R / r^3` -- one order higher in
             `1/r`, which is how a neglected oblateness or first relativistic
             correction really looks. Over a finite range of `r` this is
             **nearly degenerate with the law itself**: raising `GM` in
             `GM/r^2` mimics much of it. The prior cannot separate them, the
             recovered constant goes wrong, and the correction network learns
             nothing identifiable. This is the CMB identifiability problem
             wearing different clothes.

    `bump`   the missing term is localised -- a Gaussian in `r`, as an
             unmodelled resonance or an instrument response would be. The law
             has no way to imitate it at any value of `GM`, so the two are
             separable, and this is the regime a physics prior was built for.

    Both are shipped because the comparison IS the result: "a physics prior
    helps when the law is incomplete" is false as stated. It helps when the
    missing piece is **distinguishable from the law**.
    """

    eps: float
    noise: float
    shape: str = "bump"
    r_min: float = 1.0
    r_max: float = 6.0
    bump_centre: float = 3.0
    bump_width: float = 0.45

    def law(self, r, gm=GM_TRUE):
        """What every physics-informed arm is told."""
        return gm / np.asarray(r, float) ** 2

    def neglected(self, r, gm=GM_TRUE):
        """What none of them are told."""
        r = np.asarray(r, float)
        if self.shape == "power":
            return self.eps * gm * R_REF / r**3
        if self.shape == "bump":
            # Scaled by the law at the bump centre, so `eps` means the same
            # thing -- "this fraction of the signal" -- in both shapes.
            amp = self.eps * gm / self.bump_centre**2
            return amp * np.exp(-(((r - self.bump_centre) / self.bump_width) ** 2))
        raise ValueError(f"unknown shape {self.shape!r}")

    def truth(self, r, gm=GM_TRUE):
        return self.law(r, gm) + self.neglected(r, gm)

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        r = np.sort(rng.uniform(self.r_min, self.r_max, n))
        clean = self.truth(r)
        y = clean + rng.normal(0.0, self.noise * np.std(clean), n)
        return r, y, clean

    @property
    def neglected_fraction(self) -> float:
        """How big the missing term is, relative to the law, on average."""
        r = np.linspace(self.r_min, self.r_max, 400)
        return float(np.mean(np.abs(self.neglected(r)) / np.abs(self.law(r))))


@dataclass
class ArmResult:
    arm: str
    nrmse_in: float
    nrmse_out: float
    gm_error_pct: float = float("nan")
    n_params: int = 0
    correction_error: float = float("nan")
    history: dict = field(default_factory=dict)


def _fit_physics(sys_, r, y):
    """`curve_fit` on the law alone -- it has no way to express the rest."""
    from scipy.optimize import curve_fit

    popt, _ = curve_fit(lambda rr, gm: sys_.law(rr, gm), r, y, p0=[1.0], maxfev=20000)
    gm = float(popt[0])
    return (lambda rq: sys_.law(rq, gm)), gm, 1


def _fit_pinn(sys_, r, y, *, w_phys, epochs, seed, width=32, depth=3, lr=5e-3):
    """law(r; GM) + sd_y * NN(r), with GM trainable. The network only has the
    residual to represent, which is the whole point of this arm."""
    import torch

    from physprior.methods.neural import DTYPE, Standardiser, mlp

    torch.manual_seed(seed)
    std = Standardiser.fit(r.reshape(-1, 1), y)
    rs = torch.tensor(std.x(r.reshape(-1, 1)), dtype=DTYPE)
    rt = torch.tensor(r, dtype=DTYPE)
    yt = torch.tensor(y, dtype=DTYPE)

    net = mlp(1, width, depth)
    log_gm = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
    opt = torch.optim.Adam([*net.parameters(), log_gm], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {"epoch": [], "data": [], "phys": [], "GM": []}

    for ep in range(epochs):
        opt.zero_grad()
        corr = net(rs).squeeze(-1)
        gm = torch.exp(log_gm)
        pred = gm / rt**2 + std.sd_y * corr
        data = torch.mean(((pred - yt) / std.sd_y) ** 2)
        phys = torch.mean(corr**2)
        (data + w_phys * phys).backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            history["epoch"].append(ep)
            history["data"].append(float(data.detach()))
            history["phys"].append(float(phys.detach()))
            history["GM"].append(float(torch.exp(log_gm).detach()))

    gm_hat = float(torch.exp(log_gm).detach())

    def predict(rq):
        rq = np.asarray(rq, float).ravel()
        with torch.no_grad():
            c = net(torch.tensor(std.x(rq.reshape(-1, 1)), dtype=DTYPE)).squeeze(-1)
            return gm_hat / rq**2 + std.sd_y * c.numpy()

    def correction(rq):
        """What the network learned, in physical units -- comparable with the
        neglected term it was never shown."""
        rq = np.asarray(rq, float).ravel()
        with torch.no_grad():
            c = net(torch.tensor(std.x(rq.reshape(-1, 1)), dtype=DTYPE)).squeeze(-1)
            return std.sd_y * c.numpy()

    n_params = 1 + sum(p.numel() for p in net.parameters())
    return predict, gm_hat, n_params, correction, history


def _fit_nn(sys_, r, y, *, epochs, seed, width=32, depth=3, lr=5e-3):
    """The black box, at matched capacity: it learns the whole curve."""
    import torch

    from physprior.methods.neural import DTYPE, Standardiser, mlp

    torch.manual_seed(seed)
    std = Standardiser.fit(r.reshape(-1, 1), y)
    rs = torch.tensor(std.x(r.reshape(-1, 1)), dtype=DTYPE)
    ys = torch.tensor(std.y(y), dtype=DTYPE)
    net = mlp(1, width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {"epoch": [], "data": []}
    for ep in range(epochs):
        opt.zero_grad()
        loss = torch.mean((net(rs).squeeze(-1) - ys) ** 2)
        loss.backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            history["epoch"].append(ep)
            history["data"].append(float(loss.detach()))

    def predict(rq):
        rq = np.asarray(rq, float).ravel()
        with torch.no_grad():
            z = net(torch.tensor(std.x(rq.reshape(-1, 1)), dtype=DTYPE)).squeeze(-1)
            return std.y_inv(z.numpy())

    return predict, sum(p.numel() for p in net.parameters()), history


def run_one(
    eps: float,
    noise: float,
    shape: str = "bump",
    n_train: int = 40,
    seed: int = 11,
    w_phys: float = 1.0,
    epochs: int = 3000,
) -> list:
    """All three arms on one (eps, noise, budget), scored in and out of range.

    Out-of-range is `r` beyond the training span, which is where a prior
    either pays or does not.
    """
    sys_ = NeglectedSystem(eps=eps, noise=noise, shape=shape)
    r, y, _ = sys_.sample(n_train, seed)
    r_in = np.linspace(sys_.r_min, sys_.r_max, 200)
    r_out = np.linspace(sys_.r_max, sys_.r_max * 1.8, 200)
    truth_in, truth_out = sys_.truth(r_in), sys_.truth(r_out)
    scale_in, scale_out = np.std(truth_in), np.std(truth_out)

    out = []
    predict, gm, npar = _fit_physics(sys_, r, y)
    out.append(
        ArmResult(
            "physics",
            nrmse(truth_in, predict(r_in), scale=scale_in),
            nrmse(truth_out, predict(r_out), scale=scale_out),
            abs(gm - GM_TRUE) / GM_TRUE * 100,
            npar,
        )
    )

    predict, gm, npar, correction, hist = _fit_pinn(
        sys_, r, y, w_phys=w_phys, epochs=epochs, seed=seed
    )
    true_corr = sys_.neglected(r_in)
    denom = np.std(true_corr) or 1.0
    out.append(
        ArmResult(
            "pinn",
            nrmse(truth_in, predict(r_in), scale=scale_in),
            nrmse(truth_out, predict(r_out), scale=scale_out),
            abs(gm - GM_TRUE) / GM_TRUE * 100,
            npar,
            correction_error=float(
                np.sqrt(np.mean((correction(r_in) - true_corr) ** 2)) / denom
            ),
            history=hist,
        )
    )

    predict, npar, hist = _fit_nn(sys_, r, y, epochs=epochs, seed=seed)
    out.append(
        ArmResult(
            "nn",
            nrmse(truth_in, predict(r_in), scale=scale_in),
            nrmse(truth_out, predict(r_out), scale=scale_out),
            n_params=npar,
            history=hist,
        )
    )
    return out


def sweep(values, key: str = "eps", seeds=REPORT_SEEDS, **kw):
    """One dial at a time, over the reporting seeds."""
    import pandas as pd

    rows = []
    for v in values:
        for seed in seeds:
            for res in run_one(**{key: v, "seed": seed, **kw}):
                rows.append(
                    {
                        key: v,
                        "seed": seed,
                        "arm": res.arm,
                        "nrmse_in": res.nrmse_in,
                        "nrmse_out": res.nrmse_out,
                        "gm_error_pct": res.gm_error_pct,
                        "n_params": res.n_params,
                        "correction_error": res.correction_error,
                    }
                )
        print(f"  {key}={v} done", flush=True)
    return pd.DataFrame(rows)
