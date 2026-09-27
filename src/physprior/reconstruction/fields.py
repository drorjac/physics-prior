"""Physical fields with known truth, in 1, 2 and 3 dimensions.

Two families, both solutions of Poisson's equation

    -laplacian(u) = sum_k q_k g_w(x - c_k),

with K localized Gaussian sources of known width w and unknown positions c_k
and strengths q_k.

`box`   steady heat in the unit rod / plate / cube with Dirichlet walls. The
        wall temperature is a harmonic lift a0 + a.x, which is exact on the
        grid as well as in the continuum, so the boundary data add d + 1
        unknowns. The source is the odd image sum of the Gaussian, which makes
        the sine series below the exact solution:

            u(x) = a0 + a.x + sum_k q_k phi(x; c_k),
            phi(x; c) = sum_{n in N^d} prod_i [2 e(n_i) sin(n_i pi c_i)
                        sin(n_i pi x_i)] / (pi^2 |n|^2),
            e(n) = exp(-(n pi w)^2 / 2).

        The series is truncated at `kmax` modes per axis; the truncation
        error falls like e(kmax) and is measured in `convergence.py`. A
        second-order finite-difference solver (`solve_fd`) is the independent
        check and is also used for the mixed Dirichlet/Neumann plate of the
        model-mismatch study.

`free`  the potential of Gaussian charges in free space (electrostatic,
        gravitational, magnetic scalar potential: the same equation),
        closed form in each dimension:

            1-D  Phi = -(1/2) [r erf(r/s) + (s/sqrt(pi)) exp(-r^2/s^2)]
            2-D  Phi = -(1/2pi) [ln r + E1(r^2/2w^2)/2]
            3-D  Phi = erf(r/s) / (4 pi r),                 s = sqrt(2) w,

        which tend to the point-source Green's functions -|x|/2, -ln r/2pi
        and 1/4pi r away from the charge. A constant offset a0 is the only
        "boundary" unknown.

Units: lengths are in units of the box side, so every coordinate is
dimensionless and in [0, 1] (box) or [-0.5, 1.5] (free evaluation region).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.special import erf, exp1

from physprior.units import require

# Model constants (not physical constants): the source width and count are
# part of the physical model given to the physics-constrained methods.
SOURCE_WIDTH = 0.07
N_SOURCES = 3
# Where the truth draws its sources; the fits search a wider region.
SOURCE_REGION = (0.2, 0.8)
MIN_SEPARATION = 0.25
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant (DLMF 5.2.3)

# Series truncation per dimension. `TRUTH` is converged to round-off at the
# evaluation points; `FIT` is what the physics arm uses inside its loop,
# chosen from the truncation study (relative error below 1e-4).
KMAX_TRUTH = {1: 48, 2: 40, 3: 32}
KMAX_FIT = {1: 24, 2: 20, 3: 18}

CASES = ("box", "free")


# ---------------------------------------------------------------------------
# sine series of the box Green's function convolved with a Gaussian


def _modes(km: int) -> np.ndarray:
    return np.arange(1, km + 1, dtype=float)


def _envelope(km: int, w: float) -> np.ndarray:
    k = _modes(km)
    return 2.0 * np.exp(-0.5 * (k * np.pi * w) ** 2)


def _weights(d: int, km: int) -> np.ndarray:
    k2 = _modes(km) ** 2
    grids = np.meshgrid(*([k2] * d), indexing="ij")
    return 1.0 / (np.pi**2 * np.sum(grids, axis=0))


def _contract(A: list[np.ndarray], W: np.ndarray) -> np.ndarray:
    """sum_n prod_i A_i[p, n_i] W[n]."""
    d = len(A)
    km = A[0].shape[1]
    if d == 1:
        return A[0] @ W
    if d == 2:
        return np.sum((A[0] @ W) * A[1], axis=1)
    T = (A[2] @ W.reshape(km * km, km).T).reshape(-1, km, km)
    return np.einsum("pa,pb,pab->p", A[0], A[1], T)


_WCACHE: dict[tuple[int, int], np.ndarray] = {}


def _W(d: int, km: int) -> np.ndarray:
    key = (d, km)
    if key not in _WCACHE:
        _WCACHE[key] = _weights(d, km)
    return _WCACHE[key]


def box_green(
    x: np.ndarray, c: np.ndarray, *, w: float, km: int, grad: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    """phi(x; c) at points x (M, d), and optionally d phi / d c (M, d)."""
    x = np.atleast_2d(np.asarray(x, float))
    c = np.asarray(c, float).ravel()
    M, d = x.shape
    k = _modes(km)
    e = _envelope(km, w)
    W = _W(d, km)
    out = np.empty(M)
    gout = np.empty((M, d)) if grad else None
    for s in range(0, M, 2048):
        sx = np.sin(np.pi * x[s : s + 2048, :, None] * k)  # (m, d, km)
        A = [sx[:, i, :] * (e * np.sin(np.pi * c[i] * k)) for i in range(d)]
        out[s : s + 2048] = _contract(A, W)
        if gout is not None:
            for j in range(d):
                dA = sx[:, j, :] * (e * k * np.pi * np.cos(np.pi * c[j] * k))
                gout[s : s + 2048, j] = _contract(
                    [dA if i == j else A[i] for i in range(d)], W
                )
    return out, gout


def box_green_candidates(
    x: np.ndarray, grid1d: np.ndarray, *, w: float, km: int
) -> np.ndarray:
    """phi(x; c) for every c on the tensor grid grid1d^d, shape (M, n^d).

    The grid is separable, so the mode sum is contracted one axis at a time
    instead of evaluating n^d sources separately.
    """
    x = np.atleast_2d(np.asarray(x, float))
    M, d = x.shape
    k = _modes(km)
    e = _envelope(km, w)
    S = e[:, None] * np.sin(np.pi * np.outer(k, grid1d))  # (km, n)
    W = _W(d, km)
    n = len(grid1d)
    out = np.empty((M, n**d))
    for s in range(0, M, 256):
        sx = np.sin(np.pi * x[s : s + 256, :, None] * k)
        D = [sx[:, i, :, None] * S[None] for i in range(d)]  # (m, km, n)
        # explicit batched matmuls: np.einsum picks a far slower path here
        D0t = np.swapaxes(D[0], 1, 2)  # (m, n, km)
        if d == 1:
            B = D0t @ W
        elif d == 2:
            B = (D0t @ W) @ D[1]  # (m, x, y)
        else:
            T1 = W.reshape(km * km, km)[None] @ D[2]  # (m, a*b, z)
            T1 = T1.reshape(-1, km, km, n).transpose(0, 1, 3, 2)  # (m, a, z, b)
            T2 = T1 @ D[1][:, None]  # (m, a, z, y)
            B = (D0t @ T2.reshape(-1, km, n * n)).reshape(-1, n, n, n)
            B = B.transpose(0, 1, 3, 2)  # (m, x, y, z)
        out[s : s + 256] = B.reshape(B.shape[0], -1)
    return out


def odd_gaussian(x: np.ndarray, c: np.ndarray, w: float) -> np.ndarray:
    """The Gaussian source with its odd images across every wall of [0,1]^d.

    Product of 1-D odd, 2-periodic image sums; images beyond |m| = 2 are
    below 1e-300 for w < 0.2.
    """
    x = np.atleast_2d(np.asarray(x, float))
    out = np.ones(len(x))
    norm = 1.0 / np.sqrt(2 * np.pi * w * w)
    for i in range(x.shape[1]):
        g = np.zeros(len(x))
        for m in range(-2, 3):
            g += np.exp(-0.5 * ((x[:, i] - c[i] - 2 * m) / w) ** 2)
            g -= np.exp(-0.5 * ((x[:, i] + c[i] - 2 * m) / w) ** 2)
        out *= norm * g
    return out


def gaussian(x: np.ndarray, c: np.ndarray, w: float) -> np.ndarray:
    x = np.atleast_2d(np.asarray(x, float))
    d = x.shape[1]
    r2 = np.sum((x - np.asarray(c, float)) ** 2, axis=1)
    return np.exp(-0.5 * r2 / w**2) / (2 * np.pi * w * w) ** (d / 2)


# ---------------------------------------------------------------------------
# free-space potential of a Gaussian charge


def free_potential(r: np.ndarray, d: int, w: float) -> np.ndarray:
    """Phi(r) with -laplacian(Phi) = unit Gaussian charge of width w."""
    r = np.asarray(r, float)
    s = np.sqrt(2.0) * w
    if d == 1:
        return -0.5 * (r * erf(r / s) + s / np.sqrt(np.pi) * np.exp(-((r / s) ** 2)))
    if d == 2:
        z = r**2 / (2 * w * w)
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.log(np.where(r > 0, r, 1.0)) + 0.5 * exp1(np.where(z > 0, z, 1.0))
        # r -> 0: ln r + E1(z)/2 -> ln(sqrt(2) w) - gamma/2
        v = np.where(r > 1e-8 * w, v, np.log(s) - 0.5 * EULER_GAMMA)
        return -v / (2 * np.pi)
    if d == 3:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = erf(r / s) / np.where(r > 0, r, 1.0)
        v = np.where(r > 1e-8 * w, v, 2.0 / (s * np.sqrt(np.pi)))
        return v / (4 * np.pi)
    raise ValueError(f"d must be 1, 2 or 3, got {d}")


def free_green(
    x: np.ndarray, c: np.ndarray, *, w: float, grad: bool = False
) -> tuple[np.ndarray, np.ndarray | None]:
    x = np.atleast_2d(np.asarray(x, float))
    c = np.asarray(c, float).ravel()
    d = x.shape[1]
    val = free_potential(np.linalg.norm(x - c, axis=1), d, w)
    if not grad:
        return val, None
    # central differences on a closed form: exact to ~1e-10, and it avoids
    # the cancellation in the analytic derivative near r = 0
    h = 1e-5
    g = np.empty((len(x), d))
    for j in range(d):
        dc = np.zeros(d)
        dc[j] = h
        g[:, j] = (
            free_potential(np.linalg.norm(x - c - dc, axis=1), d, w)
            - free_potential(np.linalg.norm(x - c + dc, axis=1), d, w)
        ) / (2 * h)
    return val, g


# ---------------------------------------------------------------------------
# the physical model a fit is given


@dataclass(frozen=True)
class Geometry:
    """What the physics-constrained methods know: the PDE, the source shape,
    the boundary family and the domain. Not the source positions/strengths
    or the boundary coefficients."""

    case: str
    d: int
    w: float = SOURCE_WIDTH
    km: int = 0  # series truncation (box only); 0 -> KMAX_FIT[d]

    def __post_init__(self):
        require(self.case in CASES, f"unknown case {self.case!r}")
        require(self.d in (1, 2, 3), f"dimension must be 1-3, got {self.d}")
        require(0.0 < self.w < 0.2, f"source width {self.w} outside (0, 0.2)")

    @property
    def kmax(self) -> int:
        return self.km or KMAX_FIT[self.d]

    @property
    def n_lift(self) -> int:
        return self.d + 1 if self.case == "box" else 1

    def n_unknowns(self, K: int = N_SOURCES) -> int:
        return K * (self.d + 1) + self.n_lift

    @property
    def fit_region(self) -> tuple[float, float]:
        """Bounds for source positions in a fit (wider than the truth's)."""
        return (0.05, 0.95) if self.case == "box" else (-0.25, 1.25)

    @property
    def sensor_region(self) -> tuple[float, float]:
        return (0.0, 1.0)

    @property
    def eval_region(self) -> tuple[float, float]:
        return (0.0, 1.0) if self.case == "box" else (-0.5, 1.5)

    def green(self, x, c, grad=False):
        if self.case == "box":
            return box_green(x, c, w=self.w, km=self.kmax, grad=grad)
        return free_green(x, c, w=self.w, grad=grad)

    def candidates(self, x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Basis at x for sources on an n^d grid in the truth's source region
        widened by one step; returns (B (M, n^d), positions (n^d, d))."""
        lo, hi = SOURCE_REGION
        step = (hi - lo) / max(n - 1, 1)
        g = np.linspace(lo - 0.5 * step, hi + 0.5 * step, n)
        pos = np.stack(np.meshgrid(*([g] * self.d), indexing="ij"), -1).reshape(
            -1, self.d
        )
        if self.case == "box":
            return box_green_candidates(x, g, w=self.w, km=self.kmax), pos
        x = np.atleast_2d(x)
        r = np.linalg.norm(x[:, None, :] - pos[None], axis=2)
        return free_potential(r, self.d, self.w), pos

    def lift(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        one = np.ones((len(x), 1))
        return np.hstack([one, x]) if self.case == "box" else one


# ---------------------------------------------------------------------------
# a field with known truth


@dataclass
class Field:
    geom: Geometry
    centers: np.ndarray  # (K, d)
    q: np.ndarray  # (K,)
    lift_coef: np.ndarray  # (n_lift,)
    km_truth: int = 0
    # model-mismatch ingredients, invisible to the fitted geometry
    extra_centers: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))
    extra_q: np.ndarray = field(default_factory=lambda: np.zeros(0))
    neumann_x1: bool = False  # insulated wall at x_1 = 1 (2-D FD truth)
    _fd: object = field(default=None, repr=False)

    @property
    def d(self) -> int:
        return self.geom.d

    def value(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        require(x.shape[1] == self.d, "field: x has the wrong dimension")
        if self.neumann_x1:
            return self._fd_value(x)
        g = replace(self.geom, km=self.km_truth or KMAX_TRUTH[self.d])
        u = g.lift(x) @ self.lift_coef
        for c, q in zip(self.centers, self.q, strict=True):
            u = u + q * g.green(x, c)[0]
        for c, q in zip(self.extra_centers, self.extra_q, strict=True):
            u = u + q * g.green(x, c)[0]
        return u

    def source(self, x: np.ndarray) -> np.ndarray:
        """Right-hand side f(x) of -laplacian(u) = f."""
        src = odd_gaussian if self.geom.case == "box" else gaussian
        f = np.zeros(len(np.atleast_2d(x)))
        cs = list(self.centers) + list(self.extra_centers)
        qs = list(self.q) + list(self.extra_q)
        for c, q in zip(cs, qs, strict=True):
            f += q * src(x, c, self.geom.w)
        return f

    def _fd_value(self, x):
        from scipy.interpolate import RegularGridInterpolator

        if self._fd is None:
            n = 256
            grid, u = solve_fd(
                self.d,
                n,
                self.source,
                lambda z: self.geom.lift(z) @ self.lift_coef,
                neumann_x1=True,
            )
            self._fd = RegularGridInterpolator(
                (grid,) * self.d, u, method="cubic" if self.d > 1 else "linear"
            )
        return self._fd(np.clip(x, 0.0, 1.0))


def draw_field(
    case: str, d: int, seed: int, *, K: int = N_SOURCES, w: float = SOURCE_WIDTH
) -> Field:
    """A random field: positions, strengths and boundary data from `seed`."""
    rng = np.random.default_rng([seed, d, CASES.index(case), 0])
    lo, hi = SOURCE_REGION
    for _ in range(10_000):
        c = rng.uniform(lo, hi, size=(K, d))
        dist = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(K) * 9
        if dist.min() >= MIN_SEPARATION * (1.0 if d > 1 else 0.6):
            break
    mag = rng.uniform(0.5, 1.5, size=K)
    if case == "box":
        q = mag  # heat sources
    else:
        sign = np.ones(K)
        sign[rng.permutation(K)[: K // 2]] = -1.0  # mixed charges
        q = mag * sign
    geom = Geometry(case, d, w)
    f = Field(geom, c, q, np.zeros(geom.n_lift))
    if case == "box":
        # wall temperature on the same scale as the source-driven part
        xs = rng.uniform(0, 1, size=(512, d))
        scale = float(np.std(f.value(xs))) or 1.0
        f.lift_coef = scale * rng.uniform(-1.0, 1.0, size=d + 1)
    return f


def mismatch_field(kind: str, seed: int, d: int = 2) -> Field:
    """A box field whose physics the fitted geometry does not fully contain.

    `extra_source`  a fourth, weaker source (40% of the mean strength).
    `neumann_wall`  the wall at x_1 = 1 is insulated; the model assumes it
                    is held at the linear lift temperature like the others.
    """
    f = draw_field("box", d, seed)
    rng = np.random.default_rng([seed, d, 7, 1])
    if kind == "extra_source":
        for _ in range(10_000):
            c = rng.uniform(*SOURCE_REGION, size=d)
            if np.min(np.linalg.norm(f.centers - c, axis=1)) >= MIN_SEPARATION:
                break
        f.extra_centers = c[None]
        f.extra_q = np.array([0.4 * float(np.mean(f.q))])
    elif kind == "neumann_wall":
        f.neumann_x1 = True
    else:
        raise ValueError(f"unknown mismatch kind {kind!r}")
    return f


# ---------------------------------------------------------------------------
# finite-difference solver: the independent check, and the Neumann truth


def solve_fd(d, n, source, boundary, *, neumann_x1=False):
    """Second-order FD solution of -laplacian(u) = source on [0,1]^d.

    Nodes x_j = j/n, j = 0..n, per axis. Dirichlet value `boundary(x)` on
    every wall, except x_1 = 1 when `neumann_x1` (zero normal derivative,
    ghost-node reflection, second order). Returns (grid1d, u (n+1,)*d).
    """
    h = 1.0 / n
    g1 = np.linspace(0.0, 1.0, n + 1)
    m = n + 1
    main = np.full(m, 2.0)
    off = np.full(m - 1, -1.0)
    D = sparse.diags([off, main, off], [-1, 0, 1], format="lil")
    D1 = D.copy()
    if neumann_x1:
        D1[m - 1, m - 2] = -2.0
    D = D.tocsr() / h**2
    D1 = D1.tocsr() / h**2
    eye = sparse.identity(m, format="csr")
    A = sparse.csr_matrix((m**d, m**d))
    for i in range(d):
        mats = [eye] * d
        mats[i] = D1 if i == 0 else D
        term = mats[0]
        for mat in mats[1:]:
            term = sparse.kron(term, mat, format="csr")
        A = A + term
    X = np.stack(np.meshgrid(*([g1] * d), indexing="ij"), -1).reshape(-1, d)
    rhs = source(X)
    on_wall = np.zeros(len(X), bool)
    for i in range(d):
        on_wall |= X[:, i] == 0.0
        if not (i == 0 and neumann_x1):
            on_wall |= X[:, i] == 1.0
    idx = np.flatnonzero(on_wall)
    free = np.flatnonzero(~on_wall)
    u = np.empty(len(X))
    u[idx] = boundary(X[idx])
    A = A.tocsr()
    # eliminate the known wall values; the Dirichlet-only system that remains
    # is symmetric positive definite, so 3-D grids can use conjugate gradients
    Aff = A[free][:, free]
    b = rhs[free] - A[free][:, idx] @ u[idx]
    if d == 3 and not neumann_x1:
        from scipy.sparse.linalg import cg

        sol, info = cg(Aff, b, rtol=1e-12, atol=0.0, maxiter=20 * m)
        require(info == 0, f"FD solver: CG did not converge (info={info})")
    else:
        sol = spsolve(Aff.tocsc(), b)
    u[free] = sol
    return g1, u.reshape((m,) * d)


# ---------------------------------------------------------------------------
# sensors and evaluation grids


def sensors(
    f: Field, n_max: int, seed: int, noise_rel: float, scale: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """n_max nested random sensors and noisy readings. The first N of the
    returned arrays is the N-sensor design, so designs are nested in N.
    Returns (x, y_noisy, y_clean)."""
    require(0.0 <= noise_rel < 1.0, f"relative noise {noise_rel} outside [0, 1)")
    key = [seed, f.d, CASES.index(f.geom.case)]
    lo, hi = f.geom.sensor_region
    # separate streams for positions and noise keep both nested in n_max
    x = np.random.default_rng([*key, 1]).uniform(lo, hi, size=(n_max, f.d))
    y0 = f.value(x)
    eps = np.random.default_rng([*key, 2]).standard_normal(n_max)
    return x, y0 + noise_rel * scale * eps, y0


EVAL_POINTS = {
    ("box", 1): 201,
    ("box", 2): 51,
    ("box", 3): 19,
    ("free", 1): 401,
    ("free", 2): 61,
    ("free", 3): 25,
}


def eval_grid(geom: Geometry, n: int | None = None) -> np.ndarray:
    n = n or EVAL_POINTS[(geom.case, geom.d)]
    lo, hi = geom.eval_region
    g = np.linspace(lo, hi, n)
    return np.stack(np.meshgrid(*([g] * geom.d), indexing="ij"), -1).reshape(-1, geom.d)
