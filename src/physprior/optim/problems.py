"""The problems the optimization study runs on, and the three model families.

A `Task` is data plus physics: training points, an in-range and an
out-of-range test set scored against clean values, the closed-form law, and
(for the differential-equation tasks) the residual a PINN is trained on.

    hydrogen     real data (NIST H I levels), Bohr law E_n = R (1 - 1/n^2)
    oscillator   simulated x'' + 2 gamma x' + omega0^2 x = 0, gamma and omega0
                 unknown, analytic solution as truth
    heat         simulated u_t = D u_xx on [0, 1] with u = 0 at both ends,
                 D and two mode amplitudes unknown, separable solution as truth

A `Model` is one of three families on a task:

    physics  the closed-form law with its constants trainable (1-3 numbers)
    pinn     hydrogen: the repo's shape B, law(x; theta) + sd_y NN(x) with the
             correction penalised; oscillator/heat: the residual PINN, where
             the network is the solution and the ODE/PDE residual is the
             physics term, with the constants trainable
    nn       a plain tanh MLP on the data

Both simulations use closed-form solutions, so there is no solver whose
step size could leak into the truth. The PINN residuals use autograd
derivatives, which are exact for the network.

The curvature tools at the bottom work for any model: Hessian-vector
products by double backpropagation, the full Hessian for small models, and
Lanczos (ARPACK via scipy `eigsh`) for the top of the spectrum of large ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import torch

from physprior.methods.neural import DTYPE, mlp
from physprior.methods.pinn import ParamSet, PhysParam
from physprior.units import require

# Ground truth of the two simulations. Chosen once, before any fitting, to
# put each problem in a regime with a clear physics answer: the oscillator is
# underdamped (gamma < omega0) and shows ~2 periods in the training window;
# the heat problem's third mode decays ~9x faster than the first, so both
# modes are visible early and only the first survives out of range.
OSC_TRUTH = {"gamma": 0.2, "omega0": 2.0}
OSC_T_TRAIN = (0.0, 6.0)
OSC_T_OUT = (6.0, 10.0)
OSC_NOISE = 0.02

HEAT_TRUTH = {"D": 0.1, "a1": 1.0, "a3": 0.5}
HEAT_T_TRAIN = (0.0, 0.5)
HEAT_T_OUT = (0.5, 1.0)
HEAT_NOISE = 0.01

# Hydrogen: the range the network sees, and the levels held out inside it.
H_N_RANGE = 15
H_N_HELD_IN = 5

MODELS = ("physics", "pinn", "nn")
TASKS = ("hydrogen", "oscillator", "heat")

# The networks match the repo's default arm (width 32, depth 3, tanh).
WIDTH, DEPTH = 32, 3


@dataclass
class Task:
    name: str
    x_train: np.ndarray  # (N, d)
    y_train: np.ndarray  # (N,)
    x_in: np.ndarray  # held out, inside the training range
    y_in: np.ndarray  # clean truth there
    x_out: np.ndarray  # outside the training range
    y_out: np.ndarray
    law_t: Callable  # law_t(x (N,d) tensor, **theta) -> (N,)
    params: list[PhysParam]
    truth: dict[str, float]
    key_param: str  # the constant whose recovery is scored
    noise_sd: float
    pinn_shape: str  # "B" (law + correction) or "residual"
    x_colloc: np.ndarray | None = None  # residual PINN collocation points
    x_lo: np.ndarray = field(default_factory=lambda: np.zeros(1))
    x_hi: np.ndarray = field(default_factory=lambda: np.ones(1))
    scale: float = 1.0  # spread of the clean target, for nrmse

    @property
    def mu_y(self) -> float:
        return float(np.mean(self.y_train))

    @property
    def sd_y(self) -> float:
        return float(np.std(self.y_train)) or 1.0

    @property
    def tol(self) -> float:
        """The data loss (MSE in units of sd_y^2) a run must reach to count
        as converged: 1.5x the noise floor, plus a floor of 1e-4 (an RMS
        misfit of 1% of the spread) for noise-free data."""
        return 1.5 * (self.noise_sd / self.sd_y) ** 2 + 1e-4


# ---------------------------------------------------------------------------
# the three tasks
# ---------------------------------------------------------------------------


def _hydrogen_law_t(x, R):
    n = x[:, 0]
    return R * (1.0 - 1.0 / n**2)


@lru_cache(maxsize=1)
def hydrogen() -> Task:
    """Real NIST levels. Range n <= 15, five of those held out; n > 15 is out
    of range. The split is fixed, so seeds vary only the initialisation."""
    from physprior.constants import RYDBERG_H_CM
    from physprior.problems.quantum import hydrogen as track

    prob, _ = track.problem()
    n = prob.x[:, 0]
    in_range = np.flatnonzero(n <= H_N_RANGE)
    rng = np.random.default_rng(0)
    # n = 1 (E = 0) and the edge of the range stay in training.
    candidates = in_range[(n[in_range] > 1) & (n[in_range] < H_N_RANGE)]
    held = np.sort(rng.choice(candidates, H_N_HELD_IN, replace=False))
    train = np.setdiff1d(in_range, held)
    out = np.flatnonzero(n > H_N_RANGE)
    require(len(train) >= 8, "hydrogen: too few training levels")
    return Task(
        name="hydrogen",
        x_train=prob.x[train],
        y_train=prob.y[train],
        x_in=prob.x[held],
        y_in=prob.y[held],
        x_out=prob.x[out],
        y_out=prob.y[out],
        law_t=_hydrogen_law_t,
        params=[PhysParam("R", 1.0e5, positive=True)],
        truth={"R": RYDBERG_H_CM},
        key_param="R",
        noise_sd=0.0,
        pinn_shape="B",
        x_lo=np.array([1.0]),
        x_hi=np.array([float(H_N_RANGE)]),
        scale=float(np.std(prob.y)),
    )


def oscillator_solution(t, gamma, omega0):
    """Released from rest at x = 1:
    x(t) = exp(-gamma t) [cos(wd t) + (gamma / wd) sin(wd t)],
    wd = sqrt(omega0^2 - gamma^2).

    Evaluated with a complex wd, so the same expression is the overdamped
    solution (cosh, sinh) when gamma > omega0. A fit that starts or wanders
    into that regime then sees a finite loss instead of NaN."""
    wd = np.emath.sqrt(omega0**2 - gamma**2)
    return np.real(np.exp(-gamma * t) * (np.cos(wd * t) + gamma / wd * np.sin(wd * t)))


def _osc_law_t(x, gamma, omega0):
    t = x[:, 0].to(torch.complex128)
    wd = torch.sqrt((omega0**2 - gamma**2).to(torch.complex128))
    g = gamma.to(torch.complex128)
    z = torch.exp(-g * t) * (torch.cos(wd * t) + g / wd * torch.sin(wd * t))
    return torch.real(z)


def oscillator(n_train: int = 40, n_colloc: int = 100) -> Task:
    rng = np.random.default_rng(0)
    g = OSC_TRUTH
    t_tr = np.sort(rng.uniform(*OSC_T_TRAIN, n_train))
    y_tr = oscillator_solution(t_tr, **g) + rng.normal(0.0, OSC_NOISE, n_train)
    t_in = np.linspace(*OSC_T_TRAIN, 61)[1:-1:2]
    t_out = np.linspace(OSC_T_OUT[0], OSC_T_OUT[1], 41)[1:]
    t_all = np.linspace(OSC_T_TRAIN[0], OSC_T_OUT[1], 400)
    return Task(
        name="oscillator",
        x_train=t_tr.reshape(-1, 1),
        y_train=y_tr,
        x_in=t_in.reshape(-1, 1),
        y_in=oscillator_solution(t_in, **g),
        x_out=t_out.reshape(-1, 1),
        y_out=oscillator_solution(t_out, **g),
        law_t=_osc_law_t,
        params=[PhysParam("gamma", 0.5), PhysParam("omega0", 1.5)],
        truth=dict(g),
        key_param="gamma",
        noise_sd=OSC_NOISE,
        pinn_shape="residual",
        # Collocation covers the out-of-range window too: that is where the
        # residual PINN is supposed to earn its keep.
        x_colloc=np.linspace(OSC_T_TRAIN[0], OSC_T_OUT[1], n_colloc).reshape(-1, 1),
        x_lo=np.array([OSC_T_TRAIN[0]]),
        x_hi=np.array([OSC_T_OUT[1]]),
        scale=float(np.std(oscillator_solution(t_all, **g))),
    )


def heat_solution(x, t, D, a1, a3, lib=np):
    pi = np.pi
    return a1 * lib.exp(-D * pi**2 * t) * lib.sin(pi * x) + a3 * lib.exp(
        -9.0 * D * pi**2 * t
    ) * lib.sin(3.0 * pi * x)


def _heat_law_t(x, D, a1, a3):
    return heat_solution(x[:, 0], x[:, 1], D, a1, a3, lib=torch)


def heat(n_train: int = 60, n_colloc: int = 150) -> Task:
    rng = np.random.default_rng(0)
    h = HEAT_TRUTH

    def grid(t_lo, t_hi, n, r):
        return np.column_stack([r.uniform(0, 1, n), r.uniform(t_lo, t_hi, n)])

    x_tr = grid(*HEAT_T_TRAIN, n_train, rng)
    y_tr = heat_solution(x_tr[:, 0], x_tr[:, 1], **h) + rng.normal(
        0.0, HEAT_NOISE, n_train
    )
    x_in = grid(*HEAT_T_TRAIN, 60, rng)
    x_out = grid(*HEAT_T_OUT, 60, rng)
    x_c = grid(HEAT_T_TRAIN[0], HEAT_T_OUT[1], n_colloc, rng)
    x_all = grid(HEAT_T_TRAIN[0], HEAT_T_OUT[1], 2000, rng)
    return Task(
        name="heat",
        x_train=x_tr,
        y_train=y_tr,
        x_in=x_in,
        y_in=heat_solution(x_in[:, 0], x_in[:, 1], **h),
        x_out=x_out,
        y_out=heat_solution(x_out[:, 0], x_out[:, 1], **h),
        law_t=_heat_law_t,
        params=[PhysParam("D", 0.05), PhysParam("a1", 0.8), PhysParam("a3", 0.3)],
        truth=dict(h),
        key_param="D",
        noise_sd=HEAT_NOISE,
        pinn_shape="residual",
        x_colloc=x_c,
        x_lo=np.array([0.0, HEAT_T_TRAIN[0]]),
        x_hi=np.array([1.0, HEAT_T_OUT[1]]),
        scale=float(np.std(heat_solution(x_all[:, 0], x_all[:, 1], **h))),
    )


def get_task(name: str) -> Task:
    if name == "hydrogen":
        return hydrogen()
    if name == "oscillator":
        return oscillator()
    if name == "heat":
        return heat()
    raise ValueError(f"unknown task {name!r}")


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def _t(a: np.ndarray, grad: bool = False) -> torch.Tensor:
    return torch.tensor(np.asarray(a, float), dtype=DTYPE, requires_grad=grad)


class Model:
    """One model family on one task, with the loss split into its terms."""

    def __init__(self, task: Task, kind: str, seed: int, w_phys: float = 1.0):
        require(kind in MODELS, f"unknown model kind {kind!r}")
        self.task, self.kind, self.w_phys = task, kind, float(w_phys)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        self.net: torch.nn.Module | None = None
        self.ps: ParamSet | None = None
        d = task.x_train.shape[1]
        # Inputs scaled to [-1, 1] over the whole known domain.
        self._lo = _t(task.x_lo)
        self._span = _t(np.where(task.x_hi > task.x_lo, task.x_hi - task.x_lo, 1.0))
        if kind in ("pinn", "nn"):
            self.net = mlp(d, WIDTH, DEPTH)
        if kind in ("physics", "pinn"):
            self.ps = ParamSet(task.params)
            # The seed moves the starting constants by ~30% in log space, so
            # a failure rate over seeds means something for `physics` too.
            with torch.no_grad():
                for r in self.ps.raw:
                    r.fill_(float(rng.normal(0.0, 0.3)))
        self.x = _t(task.x_train)
        self.y = _t(task.y_train)
        self.xc = (
            _t(task.x_colloc, grad=True)
            if kind == "pinn"
            and task.pinn_shape == "residual"
            and task.x_colloc is not None
            else None
        )

    # parameters ----------------------------------------------------------

    def net_params(self) -> list[torch.Tensor]:
        return list(self.net.parameters()) if self.net is not None else []

    def phys_params(self) -> list[torch.Tensor]:
        return list(self.ps.parameters()) if self.ps is not None else []

    def parameters(self) -> list[torch.Tensor]:
        return self.net_params() + self.phys_params()

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def theta(self) -> dict[str, float]:
        return self.ps.numpy() if self.ps is not None else {}

    # prediction ----------------------------------------------------------

    def _net_out(self, x: torch.Tensor) -> torch.Tensor:
        if self.net is None:
            raise RuntimeError("this model has no network")
        xs = 2.0 * (x - self._lo) / self._span - 1.0
        return self.net(xs).squeeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.task
        if self.kind == "physics":
            return t.law_t(x, **self.ps.values())  # type: ignore[union-attr]
        if self.kind == "nn" or t.pinn_shape == "residual":
            return t.mu_y + t.sd_y * self._net_out(x)
        law = t.law_t(x, **self.ps.values())  # type: ignore[union-attr]
        return law + t.sd_y * self._net_out(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.forward(_t(x)).numpy()

    # loss terms ----------------------------------------------------------

    def data_loss(self) -> torch.Tensor:
        return torch.mean(((self.forward(self.x) - self.y) / self.task.sd_y) ** 2)

    def phys_loss(self) -> torch.Tensor:
        if self.kind != "pinn":
            return torch.zeros((), dtype=DTYPE)
        if self.task.pinn_shape == "B":
            return torch.mean(self._net_out(self.x) ** 2)
        return _residual_loss(self)

    def loss(self) -> torch.Tensor:
        if self.kind != "pinn":
            return self.data_loss()
        return self.data_loss() + self.w_phys * self.phys_loss()


def _grad(out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    (g,) = torch.autograd.grad(out, x, torch.ones_like(out), create_graph=True)
    return g


def _residual_loss(m: Model) -> torch.Tensor:
    """The ODE/PDE residual in physical units, plus the boundary conditions
    the physics fixes. Constants come from the trainable ParamSet."""
    xc = m.xc
    if xc is None or m.ps is None:
        raise RuntimeError("residual loss needs collocation points and constants")
    th = m.ps.values()
    u = m.forward(xc)
    du = _grad(u, xc)
    if m.task.name == "oscillator":
        ut = du[:, 0]
        utt = _grad(ut, xc)[:, 0]
        res = utt + 2.0 * th["gamma"] * ut + th["omega0"] ** 2 * u
        # released from rest at x = 1
        t0 = torch.zeros((1, 1), dtype=DTYPE, requires_grad=True)
        u0 = m.forward(t0)
        v0 = _grad(u0, t0)[:, 0]
        ic = (u0 - 1.0) ** 2 + v0**2
        return torch.mean(res**2) + torch.sum(ic)
    if m.task.name == "heat":
        ux, ut = du[:, 0], du[:, 1]
        uxx = _grad(ux, xc)[:, 0]
        res = ut - th["D"] * uxx
        tb = xc[:, 1:2].detach()
        edge = torch.cat(
            [
                torch.cat([torch.zeros_like(tb), tb], 1),
                torch.cat([torch.ones_like(tb), tb], 1),
            ]
        )
        bc = torch.mean(m.forward(edge) ** 2)
        return torch.mean(res**2) + bc
    raise ValueError(f"no residual for task {m.task.name!r}")


# ---------------------------------------------------------------------------
# curvature
# ---------------------------------------------------------------------------


def _flat(ts) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in ts])


def hvp_operator(loss_fn: Callable[[], torch.Tensor], params: list[torch.Tensor]):
    """Returns v -> H v for the Hessian of `loss_fn` at the current params.

    Double backpropagation: g = dL/dtheta with its graph kept, then
    d(g . v)/dtheta = H v. One HVP costs about two gradients.
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    grads = tuple(
        g if g is not None else torch.zeros_like(p)
        for g, p in zip(grads, params, strict=True)
    )
    g = _flat(grads)

    def hvp(v: np.ndarray) -> np.ndarray:
        vt = torch.tensor(np.asarray(v, float).ravel(), dtype=DTYPE)
        hv = torch.autograd.grad(g, params, vt, retain_graph=True, allow_unused=True)
        return _flat(
            [
                h if h is not None else torch.zeros_like(p)
                for h, p in zip(hv, params, strict=True)
            ]
        ).numpy()

    return hvp, g.numel()


def full_hessian(loss_fn, params) -> np.ndarray:
    hvp, n = hvp_operator(loss_fn, params)
    H = np.empty((n, n))
    e = np.zeros(n)
    for i in range(n):
        e[i] = 1.0
        H[:, i] = hvp(e)
        e[i] = 0.0
    return 0.5 * (H + H.T)


def top_eigenvalues(loss_fn, params, k: int = 10, which: str = "LA") -> np.ndarray:
    """Lanczos on the HVP operator: the k largest (`LA`) or most negative
    (`SA`) eigenvalues without ever forming H."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    hvp, n = hvp_operator(loss_fn, params)
    k = min(k, n - 1)
    op = LinearOperator((n, n), matvec=hvp, dtype=float)
    vals = eigsh(op, k=k, which=which, return_eigenvectors=False, tol=1e-8)
    return np.sort(vals)[::-1]


def spectrum_summary(eig: np.ndarray, rel_floor: float = 1e-8) -> dict:
    """Numbers that describe a Hessian spectrum at a minimum.

    An overparameterised network's Hessian is singular: most eigenvalues are
    zero up to round-off, so lambda_max / lambda_min is meaningless. Reported
    instead: the condition number over the eigenvalues above
    rel_floor * lambda_max (the directions the loss constrains at all), how
    many there are, and how many are negative (not a minimum there).
    """
    eig = np.sort(np.asarray(eig, float))[::-1]
    lmax = float(eig[0])
    pos = eig[eig > rel_floor * max(lmax, 1e-300)]
    return {
        "lambda_max": lmax,
        "lambda_min": float(eig[-1]),
        "n_eig": int(eig.size),
        "n_constrained": int(pos.size),
        "frac_constrained": float(pos.size / eig.size),
        "cond_constrained": float(pos[0] / pos[-1]) if pos.size else float("nan"),
        "n_negative": int(np.sum(eig < -rel_floor * max(lmax, 1e-300))),
        "trace": float(np.sum(eig)),
    }
