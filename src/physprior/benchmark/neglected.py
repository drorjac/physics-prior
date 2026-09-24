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


# ---------------------------------------------------------------------------
# the same question for a DIFFERENTIAL law
# ---------------------------------------------------------------------------
#
# Everything above is an algebraic relation, `y = law(x) + missing`, and the
# `pinn` arm is the law-plus-correction form. The other shape of PINN solves a
# differential equation, and the neglected-term question is sharper there:
# the missing piece is a missing FORCE, and recovering it means the network
# has learned a term of the equation of motion rather than a curve.
#
# The system is the one every physicist meets first. A pendulum obeys
#
#     theta'' = -omega^2 sin(theta)
#
# and the small-angle approximation drops everything past the linear term:
#
#     theta'' = -omega^2 theta          <- what we model
#     missing  = -omega^2 (sin theta - theta)  ~  +omega^2 theta^3 / 6
#
# The dial is the AMPLITUDE, which is a physical quantity rather than a knob:
# at 10 degrees the missing force is 0.1% of the restoring force, at 90
# degrees it is 10%. And the approximation has a measurable consequence --
# the true period grows with amplitude while the modelled one does not.


@dataclass
class NeglectedODE:
    """A pendulum, modelled as a harmonic oscillator.

    `amplitude` (radians) is the dial. The modelled law is exact as it goes to
    zero and increasingly wrong as it grows, which is what the small-angle
    approximation IS.
    """

    amplitude: float
    noise: float
    omega: float = 1.0
    t_max: float = 12.0
    shape: str = "damping"
    gamma: float = 0.06

    def missing_force(self, theta, dtheta=None):
        """What the model leaves out, as a force.

        `anharmonic`  the small-angle step, -omega^2 (sin theta - theta). A
                      function of theta alone and, over a short window,
                      **nearly degenerate with omega itself**: a pendulum at
                      amplitude A has period T(A), and a harmonic oscillator
                      can simply adopt omega = 2 pi / T(A). What is left is a
                      waveform-shape difference, which is second order. This
                      is the `power` case of the algebraic study in a
                      differential costume.

        `damping`     -2 gamma thetadot. A function of the VELOCITY, so no
                      choice of omega can imitate it -- a conservative model
                      cannot produce decay at all. This is the `bump` case:
                      distinguishable by construction.
        """
        theta = np.asarray(theta, float)
        if self.shape == "anharmonic":
            return -(self.omega**2) * (np.sin(theta) - theta)
        if self.shape == "damping":
            if dtheta is None:
                return np.zeros_like(theta)
            return -2.0 * self.gamma * np.asarray(dtheta, float)
        raise ValueError(f"unknown shape {self.shape!r}")

    def solve(self, n: int = 400):
        """The truth, by integrating the full nonlinear equation."""
        from scipy.integrate import solve_ivp

        def rhs(_t, s):
            if self.shape == "anharmonic":
                return [s[1], -(self.omega**2) * np.sin(s[0])]
            return [s[1], -(self.omega**2) * s[0] - 2.0 * self.gamma * s[1]]

        t = np.linspace(0.0, self.t_max, n)
        sol = solve_ivp(
            rhs,
            (0.0, self.t_max),
            [self.amplitude, 0.0],
            t_eval=t,
            method="DOP853",
            rtol=1e-11,
            atol=1e-13,
        )
        return t, sol.y[0]

    def sample(self, n: int, seed: int):
        rng = np.random.default_rng(seed)
        t_dense, theta_dense = self.solve(2000)
        t = np.sort(rng.uniform(0.0, self.t_max, n))
        theta = np.interp(t, t_dense, theta_dense)
        return t, theta + rng.normal(0.0, self.noise * np.std(theta_dense), n)

    @property
    def missing_fraction(self) -> float:
        """The missing force as a fraction of the restoring force, at peak."""
        a = self.amplitude
        if self.shape == "damping":
            # at peak speed |thetadot| ~ omega * a
            return float(
                2.0 * self.gamma * self.omega * abs(a) / (self.omega**2 * abs(a) or 1.0)
            )
        return float(abs(self.missing_force(a)) / (self.omega**2 * abs(a) or 1.0))

    @property
    def true_period(self) -> float:
        """The real period, which grows with amplitude -- the observable
        consequence of the term being dropped."""
        from scipy.special import ellipk

        return float(4.0 / self.omega * ellipk(np.sin(self.amplitude / 2.0) ** 2))


def _ode_fit_physics(sys_, t, theta):
    """Fit the harmonic solution. It cannot bend, so it cannot be right."""
    from scipy.optimize import curve_fit

    def model(tt, omega, a):
        return a * np.cos(omega * tt)

    popt, _ = curve_fit(model, t, theta, p0=[1.0, sys_.amplitude], maxfev=40000)
    omega = float(popt[0])
    return (lambda tq: model(np.asarray(tq, float), *popt)), omega


def _ode_fit_pinn(
    sys_,
    t,
    theta,
    *,
    w_phys,
    w_res=1.0,
    epochs=4000,
    seed=0,
    width=32,
    depth=3,
    lr=5e-3,
    n_collocation=256,
):
    """Residual PINN with a LEARNED FORCE correction.

        theta'' + omega^2 theta - C(theta) = 0

    `omega` is trainable and `C` is a network of theta -- not of t. That
    matters: a correction in `t` is a curve, while a correction in `theta` is
    a term of the equation of motion, and only the second can be compared
    against the force that was dropped.
    """
    import torch

    from physprior.methods.neural import DTYPE, mlp

    torch.manual_seed(seed)
    scale = float(np.std(theta)) or 1.0
    t_t = torch.tensor(t, dtype=DTYPE)
    y_t = torch.tensor(theta, dtype=DTYPE)
    t_c = torch.linspace(0.0, sys_.t_max, n_collocation, dtype=DTYPE)

    net = mlp(1, width, depth)  # theta(t)
    corr_raw = mlp(2, width, depth)  # C(theta), the missing force
    log_w = torch.nn.Parameter(torch.zeros((), dtype=DTYPE))
    opt = torch.optim.Adam([*net.parameters(), *corr_raw.parameters(), log_w], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {"epoch": [], "data": [], "phys": [], "GM": []}

    def theta_of(tt):
        return net(tt.unsqueeze(-1)).squeeze(-1)

    def correction(th, dth=None):
        """C(theta), forced ODD by construction.

        The dropped force is -omega^2 (sin theta - theta), which is odd, and
        a restoring force must vanish at theta = 0 on symmetry grounds alone.
        Antisymmetrising the network imposes both exactly, at every step, for
        free -- and without it the correction wanders near the origin where
        the data constrains it least.
        """
        # A missing force may depend on velocity as well as position --
        # damping does, and no function of theta alone can express it. Both
        # go in, and the pair is antisymmetrised, because a force that does
        # not vanish at rest at the origin is not a correction to this
        # equation.
        d = torch.zeros_like(th) if dth is None else dth
        u = torch.stack([th, d], dim=-1)
        return (corr_raw(u) - corr_raw(-u)).squeeze(-1) / 2.0

    for ep in range(epochs):
        opt.zero_grad()
        omega = torch.exp(log_w)
        data = torch.mean(((theta_of(t_t) - y_t) / scale) ** 2)

        tc = t_c.clone().requires_grad_(True)
        th = theta_of(tc)
        dth = torch.autograd.grad(th.sum(), tc, create_graph=True)[0]
        d2th = torch.autograd.grad(dth.sum(), tc, create_graph=True)[0]
        c = correction(th, dth)
        residual = d2th + omega**2 * th - c
        phys = torch.mean(c**2)
        (data + w_res * torch.mean(residual**2) + w_phys * phys).backward()
        opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            history["epoch"].append(ep)
            history["data"].append(float(data.detach()))
            history["phys"].append(float(phys.detach()))
            history["GM"].append(float(torch.exp(log_w).detach()))

    omega_hat = float(torch.exp(log_w).detach())

    def predict(tq):
        with torch.no_grad():
            return theta_of(
                torch.tensor(np.asarray(tq, float).ravel(), dtype=DTYPE)
            ).numpy()

    def force(theta_q, dtheta_q=None):
        with torch.no_grad():
            q = torch.tensor(np.asarray(theta_q, float).ravel(), dtype=DTYPE)
            d = (
                torch.zeros_like(q)
                if dtheta_q is None
                else torch.tensor(np.asarray(dtheta_q, float).ravel(), dtype=DTYPE)
            )
            return correction(q, d).numpy()

    n_params = 1 + sum(p.numel() for p in [*net.parameters(), *corr_raw.parameters()])
    return predict, omega_hat, n_params, force, history


def _ode_fit_nn(sys_, t, theta, *, epochs=4000, seed=0, width=32, depth=3, lr=5e-3):
    import torch

    from physprior.methods.neural import DTYPE, Standardiser, mlp

    torch.manual_seed(seed)
    std = Standardiser.fit(t.reshape(-1, 1), theta)
    ts = torch.tensor(std.x(t.reshape(-1, 1)), dtype=DTYPE)
    ys = torch.tensor(std.y(theta), dtype=DTYPE)
    net = mlp(1, width, depth)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for _ in range(epochs):
        opt.zero_grad()
        torch.mean((net(ts).squeeze(-1) - ys) ** 2).backward()
        opt.step()
        sched.step()

    def predict(tq):
        tq = np.asarray(tq, float).ravel()
        with torch.no_grad():
            z = net(torch.tensor(std.x(tq.reshape(-1, 1)), dtype=DTYPE)).squeeze(-1)
            return std.y_inv(z.numpy())

    return predict, sum(p.numel() for p in net.parameters())


def run_ode(
    amplitude: float,
    noise: float = 0.02,
    shape: str = "damping",
    n_train: int = 60,
    seed: int = 11,
    w_phys: float = 1e-3,
    epochs: int = 3000,
) -> list:
    """All three arms on the pendulum, scored against the true trajectory."""
    sys_ = NeglectedODE(amplitude=amplitude, noise=noise, shape=shape)
    t, theta = sys_.sample(n_train, seed)
    t_dense, truth = sys_.solve(400)
    scale = float(np.std(truth))

    out = []
    predict, omega = _ode_fit_physics(sys_, t, theta)
    out.append(
        ArmResult(
            "physics",
            nrmse(truth, predict(t_dense), scale=scale),
            float("nan"),
            abs(omega - sys_.omega) / sys_.omega * 100,
            2,
        )
    )

    predict, omega, npar, force, hist = _ode_fit_pinn(
        sys_, t, theta, w_phys=w_phys, epochs=epochs, seed=seed
    )
    # Score the learned force along the ACTUAL trajectory, because that is
    # the only place the data constrained it -- and a damping term cannot be
    # scored on a theta grid at all, since it depends on the velocity.
    t_traj, theta_traj = sys_.solve(400)
    dtheta_traj = np.gradient(theta_traj, t_traj)
    grid, grid_d = theta_traj, dtheta_traj
    true_force = sys_.missing_force(grid, grid_d)
    denom = np.std(true_force) or 1.0
    out.append(
        ArmResult(
            "pinn",
            nrmse(truth, predict(t_dense), scale=scale),
            float("nan"),
            abs(omega - sys_.omega) / sys_.omega * 100,
            npar,
            correction_error=float(
                np.sqrt(np.mean((force(grid) - true_force) ** 2)) / denom
            ),
            history=hist,
        )
    )

    predict, npar = _ode_fit_nn(sys_, t, theta, epochs=epochs, seed=seed)
    out.append(
        ArmResult(
            "nn",
            nrmse(truth, predict(t_dense), scale=scale),
            float("nan"),
            n_params=npar,
        )
    )
    return out
