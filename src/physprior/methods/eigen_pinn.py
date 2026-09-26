"""Learning a wave function: the Schrodinger equation as a PINN.

Everything else in this package fits a curve to data. This module has **no
data at all**: the only supervision is the physics. Given a potential it
learns the pair

    -1/2 psi''(x) + V(x) psi(x) = E psi(x),     psi(boundary) = 0

for `psi` AND `E` together, and is checked against the same problem solved by
direct diagonalisation in `physprior.problems.quantum.schrodinger`.

## The four things that make this hard, and what is done about each

**1. psi = 0 solves it.** The residual vanishes identically for the zero
function, so a naive residual loss collapses to it. Two ways out are
implemented, because the comparison is the point:

  `rayleigh`  minimise the Rayleigh quotient E[psi] = <psi|H|psi> / <psi|psi>.
              Scale-invariant by construction, so psi = 0 is not a minimum --
              it is not even in the domain. Returns the GROUND state, and E
              is an output rather than a parameter.
  `residual`  minimise ||H psi - E psi||^2 with E a trainable parameter, and
              divide by <psi|psi> so the scale gauge cannot shrink the loss.

**2. Boundary conditions are a hard constraint, not a penalty.** The network
output is multiplied by a function that vanishes at the walls,

    psi(x) = (x - a)(b - x) * NN(x)

so every candidate satisfies psi(a) = psi(b) = 0 exactly, at every step of
training, and there is no boundary weight to tune. A penalty term would make
the constraint approximate and add a hyperparameter whose value nobody can
justify.

**3. Excited states need orthogonality, and the weight must dominate the
level spacing.** Both objectives are minimised by the ground state, so state
`k` is found by penalising overlap with the `k` already found:
`sum_j <psi|psi_j>^2 / <psi|psi>`, which is zero exactly when psi is
orthogonal to all of them.

The weight is not a free knob. Falling back to the ground state costs
`w * overlap^2 = w`, and climbing to the next level costs `E_1 - E_0`, so any
`w` below the gap makes collapse the cheaper option -- with `w = 10` on the
infinite well, whose first gap is 14.8, level 2 duly came back as a copy of
level 1 with overlap 1.0000. `w_ortho` is therefore scaled by the energies
already found rather than used raw, and `overlap()` is on `EigenState` so the
check is one line for a caller.

**4. Normalisation is done in the loss, not afterwards.** Every quantity is a
ratio of inner products, so the network is free to choose any scale and the
optimiser never spends capacity on one.

This is the textbook PINN of Raissi et al. (2019) applied to an eigenvalue
problem rather than an initial-value one, and it is deliberately the SLOWEST
way to get these numbers: `solve_1d` diagonalises a tridiagonal matrix in
milliseconds and is more accurate. The point is not to beat it. The point is
that the same machinery extends to problems where no such matrix exists --
and that having the exact answer available is what makes the failure modes
measurable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .base import Fit, set_seed
from .neural import DTYPE, mlp


@dataclass
class EigenState:
    """One learned eigenpair, with the diagnostics that say whether to trust it."""

    level: int
    energy: float
    x: np.ndarray
    psi: np.ndarray
    history: dict = field(default_factory=dict)
    residual_rms: float = float("nan")
    seconds: float = 0.0
    method: str = "rayleigh"
    # Set by `fit_spectrum`: the largest overlap with any lower state. A
    # level that came back as a copy of one below it has not converged,
    # whatever its residual says.
    max_overlap: float = 0.0
    converged: bool = True
    attempts: int = 1

    def normalised(self) -> np.ndarray:
        norm = np.sqrt(np.trapezoid(self.psi**2, self.x))
        return self.psi / (norm or 1.0)

    def overlap(self, other: EigenState) -> float:
        """|<psi_i|psi_j>|, which should be ~0 between different levels."""
        a, b = self.normalised(), other.normalised()
        return float(abs(np.trapezoid(a * b, self.x)))


class WaveFunction(torch.nn.Module):
    """psi(x) = envelope(x) * NN(x), with the boundary condition built in.

    `tanh`, never ReLU: the loss differentiates the output twice, and a ReLU
    network has zero second derivative almost everywhere.
    """

    # Registered buffers are Tensors, but a type checker only learns that
    # from an annotation -- without these it reads `self.a` as Tensor | Module.
    a: torch.Tensor
    b: torch.Tensor

    def __init__(self, x_min: float, x_max: float, width: int = 64, depth: int = 3):
        super().__init__()
        self.net = mlp(1, width, depth)
        self.register_buffer("a", torch.tensor(float(x_min), dtype=DTYPE))
        self.register_buffer("b", torch.tensor(float(x_max), dtype=DTYPE))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (x - self.a) * (self.b - x)
        return envelope * self.net(x.unsqueeze(-1)).squeeze(-1)


# Two states overlapping by more than this are the same state.
COLLAPSE_TOL = 0.1


def _second_derivative(psi: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """psi'' by automatic differentiation, twice. No finite differences."""
    dpsi = torch.autograd.grad(psi.sum(), x, create_graph=True)[0]
    return torch.autograd.grad(dpsi.sum(), x, create_graph=True)[0]


def _inner(f: torch.Tensor, g: torch.Tensor, dx: float) -> torch.Tensor:
    return torch.sum(f * g) * dx


def fit_eigen_pinn(
    potential,
    x_min: float,
    x_max: float,
    *,
    level: int = 0,
    lower: tuple = (),
    method: str = "rayleigh",
    n_collocation: int = 512,
    width: int = 64,
    depth: int = 3,
    epochs: int = 4000,
    lr: float = 3e-3,
    seed: int = 0,
    w_ortho: float = 10.0,  # scaled by the energies already found -- see above
    energy_init: float = 1.0,
    record_every: int = 0,
) -> EigenState:
    """Learn one eigenpair of `-1/2 psi'' + V psi = E psi` on `[x_min, x_max]`.

    `lower` holds the `EigenState`s already found; the loss is made orthogonal
    to each, which is what selects an excited state rather than the ground
    state for `level > 0`.
    """
    if method not in ("rayleigh", "residual"):
        raise ValueError(f"unknown method {method!r}")
    t0 = time.time()
    set_seed(seed)

    x = torch.linspace(x_min, x_max, n_collocation, dtype=DTYPE)
    dx = float((x_max - x_min) / (n_collocation - 1))
    v = torch.tensor(np.asarray(potential(x.numpy()), float), dtype=DTYPE)

    net = WaveFunction(x_min, x_max, width, depth)
    params = list(net.parameters())
    log_e = None
    if method == "residual":
        log_e = torch.nn.Parameter(torch.tensor(float(energy_init), dtype=DTYPE))
        params = [*params, log_e]

    lower_psi = [
        torch.tensor(np.interp(x.numpy(), s.x, s.normalised()), dtype=DTYPE)
        for s in lower
    ]
    # The orthogonality penalty has to outweigh the energy it costs to leave
    # the ground state, and that cost is set by the spectrum, not by taste.
    # Scaling by the largest energy found so far is enough: the gaps of a
    # bound spectrum are of the same order as the levels themselves.
    ortho_weight = w_ortho * (1.0 + max((abs(s.energy) for s in lower), default=0.0))

    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {"epoch": [], "loss": [], "energy": []}

    for ep in range(epochs):
        opt.zero_grad()
        xc = x.clone().requires_grad_(True)
        psi = net(xc)
        d2 = _second_derivative(psi, xc)
        h_psi = -0.5 * d2 + v * psi

        norm = _inner(psi, psi, dx)
        if method == "rayleigh" or log_e is None:
            energy_t = _inner(psi, h_psi, dx) / norm
            loss = energy_t
        else:
            energy_t = log_e
            residual_t = h_psi - energy_t * psi
            loss = _inner(residual_t, residual_t, dx) / norm

        for p_low in lower_psi:
            loss = loss + ortho_weight * _inner(psi, p_low, dx) ** 2 / norm

        loss.backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["loss"].append(float(loss.detach()))
            history["energy"].append(float(energy_t.detach()))

    with torch.no_grad():
        psi_np = net(x).numpy()
    xc = x.clone().requires_grad_(True)
    psi_t = net(xc)
    h_psi = -0.5 * _second_derivative(psi_t, xc) + v * psi_t
    norm = _inner(psi_t, psi_t, dx)
    final_e = float((_inner(psi_t, h_psi, dx) / norm).detach())
    residual = (h_psi - final_e * psi_t).detach()
    rms = float(torch.sqrt(_inner(residual, residual, dx) / norm.detach()))

    return EigenState(
        level=level,
        energy=final_e,
        x=x.numpy(),
        psi=psi_np,
        history=history,
        residual_rms=rms,
        seconds=time.time() - t0,
        method=method,
    )


def fit_spectrum(
    potential,
    x_min: float,
    x_max: float,
    n_levels: int = 3,
    retries: int = 1,
    **kw,
) -> list:
    """The lowest `n_levels` eigenpairs, found one at a time.

    Each level is made orthogonal to the ones below it, so errors ACCUMULATE
    up the ladder: an imperfect ground state is an imperfect constraint on
    the first excited state.

    **High levels collapse, and the collapse is marked rather than hidden.**
    Around the fourth level of the infinite well the optimiser stops finding
    the state even though the objective still prefers it -- falling back
    costs the orthogonality penalty, which by then far exceeds the level
    itself. It is the network that cannot get there: reshaping one lobe into
    four means crossing a barrier in function space, and networks learn low
    frequencies long before high ones. This is spectral bias, the best-known
    failure mode of PINNs, showing up in a problem where the right answer is
    known to nine digits.

    So `converged` is set from the overlap with the lower states, a failed
    level is retried from a different seed, and a level that still comes back
    as a copy is returned WITH `converged = False` rather than silently. The
    project's rule is that a parameter pinned to its bound has not converged
    whatever the optimiser reports; this is the same rule for an eigenstate.
    """
    seed = kw.pop("seed", 0)
    states: list = []
    for level in range(n_levels):
        best = None
        for attempt in range(retries + 1):
            cand = fit_eigen_pinn(
                potential,
                x_min,
                x_max,
                level=level,
                lower=tuple(states),
                seed=seed + 101 * attempt,
                **kw,
            )
            cand.max_overlap = max((cand.overlap(s) for s in states), default=0.0)
            cand.converged = cand.max_overlap <= COLLAPSE_TOL
            cand.attempts = attempt + 1
            if best is None or cand.max_overlap < best.max_overlap:
                best = cand
            if cand.converged:
                break
        states.append(best)
    return states


def as_fit(state: EigenState, name: str = "eigen_pinn") -> Fit:
    """Wrap a learned state in the package's common `Fit` type."""

    def predict(xq: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(xq, float).ravel(), state.x, state.normalised())

    return Fit(
        name=name,
        predict=predict,
        params={"E": state.energy},
        expression="-1/2 psi'' + V psi = E psi  [PINN, boundary hard-constrained]",
        seconds=state.seconds,
        extra={"residual_rms": state.residual_rms, "method": state.method},
    )


# ---------------------------------------------------------------------------
# the inverse problem: recover the potential from a measured spectrum
# ---------------------------------------------------------------------------


@dataclass
class InversePotential:
    """A potential recovered from a spectrum, and what it cost."""

    x: np.ndarray
    v: np.ndarray  # the recovered V(x)
    psi: np.ndarray  # (n_levels, n_grid) the states that came with it
    energies_target: np.ndarray
    energies_achieved: np.ndarray
    symmetric: bool
    seconds: float = 0.0
    history: dict = field(default_factory=dict)

    @property
    def spectrum_error(self) -> float:
        """Max relative error on the eigenvalues it was asked to reproduce."""
        t = np.asarray(self.energies_target, float)
        return float(np.max(np.abs((self.energies_achieved - t) / t)))

    def error_against(self, truth) -> float:
        """Max |V_hat - V_true| over the region the states actually occupy.

        Scored where the wave functions have support, because a spectrum
        cannot constrain the potential where no state visits: out in the
        classically forbidden tails `V` is free to be almost anything, and a
        global norm would report that freedom as failure.
        """
        v_true = np.asarray(truth(self.x), float)
        weight = np.sum(self.psi**2, axis=0)
        mask = weight > 0.01 * weight.max()
        return float(np.max(np.abs(self.v[mask] - v_true[mask])))


class PotentialNet(torch.nn.Module):
    """V(x), optionally forced to be even.

    Symmetry is not a convenience here. **One spectrum does not determine a
    one-dimensional potential** -- the Borg-Marchenko theorem says two
    spectra are needed in general, and the classic "can one hear the shape of
    a drum?" is the same question. For a SYMMETRIC potential one spectrum is
    enough, so `symmetric=True` buys identifiability rather than accuracy,
    and it is imposed architecturally by evaluating at |x - centre|.
    """

    def __init__(self, x_min: float, x_max: float, symmetric: bool, width=64, depth=3):
        super().__init__()
        self.net = mlp(1, width, depth)
        self.symmetric = symmetric
        self.centre = 0.5 * (x_min + x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = torch.abs(x - self.centre) if self.symmetric else x
        return self.net(u.unsqueeze(-1)).squeeze(-1)


def fit_inverse_potential(
    energies,
    x_min: float,
    x_max: float,
    *,
    symmetric: bool = True,
    method: str = "diagonalise",
    n_collocation: int = 512,
    width: int = 64,
    depth: int = 3,
    epochs: int = 6000,
    lr: float = 3e-3,
    seed: int = 0,
    w_ortho: float = 1.0,
    w_smooth: float = 1e-3,
    record_every: int = 0,
) -> InversePotential:
    """Recover `V(x)` from the eigenvalues it is supposed to produce.

    This is the inverse of `fit_eigen_pinn`, and the one worth caring about:
    the forward problem is solved better by a tridiagonal matrix, while this
    one has no such solver at all.

    The eigenvalues are **fixed to the observed values** rather than computed
    by a Rayleigh quotient, and `V` and every `psi_k` are moved until

        -1/2 psi_k'' + V psi_k - E_k psi_k = 0

    holds for all of them at once. Fixing E is what makes the problem
    well-posed: an arbitrary function has a Rayleigh quotient too, so
    matching a quotient to a number constrains far less than requiring the
    residual to vanish at that number.

    Two gauges, handled differently:

      * `V -> V + c` shifts every eigenvalue by `c`. Because the target
        eigenvalues are absolute, the data fixes this one -- nothing to do.
      * `psi_k -> a psi_k` is free, so every term is a ratio of inner
        products, as in the forward problem.

    **The problem is underdetermined, and `w_smooth` is what makes it
    tractable.** Every term of the residual is proportional to `psi`, so
    where no state has support the data says nothing at all about `V` -- and
    an unconstrained network duly puts a bump out in the tail, which
    manufactures spurious bound states and corrupts the spectrum the
    potential actually has. With three levels and a 512-point grid the fit
    reproduced the first two eigenvalues and invented a degenerate pair for
    the third.

    `w_smooth` penalises `V''`, which is Tikhonov regularisation: it does not
    add information, it states a preference for the smoothest potential
    consistent with the data. That is a prior, and it must be reported
    as one. The alternative fixes are more levels or a parametric form for
    `V`, and both are better when available.
    """
    if method not in ("residual", "diagonalise"):
        raise ValueError(f"unknown method {method!r}")
    t0 = time.time()
    set_seed(seed)
    e_target = np.asarray(energies, float).ravel()
    n_levels = len(e_target)

    if method == "diagonalise":
        return _inverse_by_diagonalising(
            e_target,
            x_min,
            x_max,
            symmetric=symmetric,
            n_grid=n_collocation,
            width=width,
            depth=depth,
            epochs=epochs,
            lr=lr,
            w_smooth=w_smooth,
            record_every=record_every,
            t0=t0,
        )

    x = torch.linspace(x_min, x_max, n_collocation, dtype=DTYPE)
    dx = float((x_max - x_min) / (n_collocation - 1))
    e_t = torch.tensor(e_target, dtype=DTYPE)
    scale = float(np.max(np.abs(e_target))) or 1.0

    v_net = PotentialNet(x_min, x_max, symmetric, width, depth)
    # One network per state: the states have different numbers of nodes and
    # sharing a trunk between them mostly shares the wrong features.
    psi_nets = torch.nn.ModuleList(
        [WaveFunction(x_min, x_max, width, depth) for _ in range(n_levels)]
    )
    opt = torch.optim.Adam(
        list(v_net.parameters()) + list(psi_nets.parameters()), lr=lr
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {
        "epoch": [],
        "loss": [],
        "residual": [],
        "ortho": [],
        "smooth": [],
    }

    for ep in range(epochs):
        opt.zero_grad()
        xc = x.clone().requires_grad_(True)
        v = v_net(xc)
        states = []
        residual_total = torch.zeros((), dtype=DTYPE)
        for k in range(n_levels):
            psi = psi_nets[k](xc)
            d2 = _second_derivative(psi, xc)
            r = -0.5 * d2 + v * psi - e_t[k] * psi
            residual_total = residual_total + _inner(r, r, dx) / (
                _inner(psi, psi, dx) * scale**2
            )
            states.append(psi)

        ortho = torch.zeros((), dtype=DTYPE)
        for i in range(n_levels):
            for j in range(i + 1, n_levels):
                ortho = ortho + _inner(states[i], states[j], dx) ** 2 / (
                    _inner(states[i], states[i], dx) * _inner(states[j], states[j], dx)
                )

        # Tikhonov on V'': the data cannot see V where no state lives, so
        # something has to choose, and "smooth" is the assumption
        # declared here. Scaled by the spectrum so it means the same thing at any
        # energy scale.
        d2v = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        smooth = _inner(d2v, d2v, dx) / scale**2
        loss = residual_total + w_ortho * ortho + w_smooth * smooth
        loss.backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["loss"].append(float(loss.detach()))
            history["residual"].append(float(residual_total.detach()))
            history["ortho"].append(float(ortho.detach()))
            history["smooth"].append(float(smooth.detach()))

    with torch.no_grad():
        v_out = v_net(x).numpy()
        psi_out = np.stack([net(x).numpy() for net in psi_nets])

    # What eigenvalues does the recovered potential actually have? Answer with
    # the independent solver, not with the network that produced it.
    achieved = _eigenvalues_of(v_out, x.numpy(), n_levels)
    return InversePotential(
        x=x.numpy(),
        v=v_out,
        psi=psi_out,
        energies_target=e_target,
        energies_achieved=achieved,
        symmetric=symmetric,
        seconds=time.time() - t0,
        history=history,
    )


def _eigenvalues_of(v: np.ndarray, x: np.ndarray, n_levels: int) -> np.ndarray:
    """Diagonalise the recovered potential, as an independent check.

    Deliberately NOT the network's own estimate. A PINN's internal
    diagnostics cannot tell you whether it is right, so the spectrum it
    claims is verified by the method it was supposed to replace.
    """
    from scipy.linalg import eigh_tridiagonal

    dx = float(x[1] - x[0])
    interior = slice(1, -1)
    d = 1.0 / dx**2 + v[interior]
    e = -0.5 / dx**2 * np.ones(len(d) - 1)
    vals = eigh_tridiagonal(
        d, e, select="i", select_range=(0, n_levels - 1), eigvals_only=True
    )
    return np.asarray(vals, float)


def _tridiagonal_eigenvalues(v: torch.Tensor, dx: float, n_levels: int):
    """The lowest `n_levels` eigenvalues of -1/2 d2/dx2 + V, differentiably.

    `torch.linalg.eigvalsh` is differentiable, so a gradient flows from the
    eigenvalues back to `V`. That is the whole trick: the forward problem is
    a matrix eigendecomposition and the matrix is a differentiable function
    of the unknown, so there is no need for a network to represent the states
    at all.
    """
    n = v.shape[0] - 2
    main = torch.diag(1.0 / dx**2 + v[1:-1])
    off = torch.diag(-0.5 / dx**2 * torch.ones(n - 1, dtype=v.dtype), 1)
    h = main + off + off.T
    return torch.linalg.eigvalsh(h)[:n_levels]


def _inverse_by_diagonalising(
    e_target,
    x_min,
    x_max,
    *,
    symmetric,
    n_grid,
    width,
    depth,
    epochs,
    lr,
    w_smooth,
    record_every,
    t0,
):
    """Recover V by differentiating through the eigensolver itself.

    The PINN formulation has to represent every wave function with its own
    network and only approximately satisfies the eigenvalue equation, so the
    eigenvalues it reports are not quite the eigenvalues the recovered
    potential has. Here they are, exactly, by construction -- there is
    nothing left to approximate on the forward side, and every bit of the
    optimisation goes into `V`.

    **When a differentiable forward model exists, this beats a PINN.** The
    PINN earns its place when one does not: an unmeshable geometry, a
    potential known only pointwise, a solver that is not differentiable.
    Keeping both here is what lets that claim be a measurement rather than an
    opinion.
    """
    x = torch.linspace(x_min, x_max, n_grid, dtype=DTYPE)
    dx = float((x_max - x_min) / (n_grid - 1))
    target = torch.tensor(np.asarray(e_target, float), dtype=DTYPE)
    scale = float(np.max(np.abs(e_target))) or 1.0

    v_net = PotentialNet(x_min, x_max, symmetric, width, depth)
    opt = torch.optim.Adam(v_net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    history: dict[str, list] = {"epoch": [], "loss": [], "residual": [], "smooth": []}

    for ep in range(epochs):
        opt.zero_grad()
        v = v_net(x)
        vals = _tridiagonal_eigenvalues(v, dx, len(target))
        match = torch.mean(((vals - target) / scale) ** 2)
        d2v = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        smooth = torch.mean(d2v**2) / scale**2
        loss = match + w_smooth * smooth
        loss.backward()
        opt.step()
        sched.step()
        if record_every and (ep % record_every == 0 or ep == epochs - 1):
            history["epoch"].append(ep)
            history["loss"].append(float(loss.detach()))
            history["residual"].append(float(match.detach()))
            history["smooth"].append(float(smooth.detach()))

    with torch.no_grad():
        v_out = v_net(x).numpy()
    achieved = _eigenvalues_of(v_out, x.numpy(), len(target))
    psi = _eigenvectors_of(v_out, x.numpy(), len(target))
    return InversePotential(
        x=x.numpy(),
        v=v_out,
        psi=psi,
        energies_target=np.asarray(e_target, float),
        energies_achieved=achieved,
        symmetric=symmetric,
        seconds=time.time() - t0,
        history=history,
    )


def _eigenvectors_of(v: np.ndarray, x: np.ndarray, n_levels: int) -> np.ndarray:
    from scipy.linalg import eigh_tridiagonal

    dx = float(x[1] - x[0])
    d = 1.0 / dx**2 + v[1:-1]
    e = -0.5 / dx**2 * np.ones(len(d) - 1)
    _, vecs = eigh_tridiagonal(d, e, select="i", select_range=(0, n_levels - 1))
    psi = np.zeros((n_levels, len(x)))
    psi[:, 1:-1] = vecs.T
    return psi / np.sqrt(np.sum(psi**2, axis=1, keepdims=True) * dx)
