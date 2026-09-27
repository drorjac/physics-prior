"""Radio propagation on a floor plan: the 2-D Helmholtz equation, solved.

The field of a time-harmonic transmitter, u(x, y) exp(-i w t), obeys

    laplacian(u) + k^2 n(x, y)^2 u = -f(x, y)

with k = 2 pi / lambda the free-space wavenumber, n the complex refractive
index of whatever fills the point (air, or a lossy wall), and f the source.
Lengths are in wavelengths throughout, so k = 2 pi. The scene is labelled for
2.4 GHz (lambda = 12.5 cm) only to pick the wall material; nothing else
depends on the frequency.

Discretisation. Second-order central differences on a uniform grid of
spacing h = 1 / ppw (points per wavelength). The domain is closed by a
perfectly matched layer: complex coordinate stretching x -> x (1 + i a
(d/L)^2) inside a layer of width L, which absorbs outgoing waves at every
angle of incidence. The operator is written in the symmetric form

    d/dx((s_y/s_x) du/dx) + d/dy((s_x/s_y) du/dy) + k^2 n^2 s_x s_y u
        = -s_x s_y f

with u = 0 at the outer edge of the layer, and solved by sparse LU.

Source. A normalised Gaussian of width SOURCE_SIGMA rather than a point:
a 2-D point source has a logarithmic singularity, so its field at the source
would never converge with h. Beyond a few widths the Gaussian's field is
exactly the point source's times exp(-k^2 sigma^2 / 2), so the analytic
free-space answer is still known:

    u(r) = (i/4) H0^(1)(k r) exp(-k^2 sigma^2 / 2)

What a receiver measures. The raw |u|^2 has multipath fading on the scale of
half a wavelength. A received-signal-strength reading is an average over
that fading (over frequency, time or antenna diversity); here it is modelled
as the mean of |u|^2 over a disk of radius AVG_RADIUS, the "local mean
power". That is the field the arms reconstruct. The small-scale fading is
kept for the figures.

Free-space exponent. In two dimensions the field of a line source spreads
cylindrically: |u|^2 ~ 1/r, so the log-distance path-loss exponent is n = 1,
not the n = 2 of a point source in three dimensions. The law the arms fit is
the same law; the value it should return in free space is 1.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
from scipy.signal import fftconvolve
from scipy.sparse.linalg import splu
from scipy.special import hankel1

from physprior.units import require, require_range

K0 = 2.0 * np.pi  # wavenumber in units of 1/lambda

# Nominal carrier, used ONLY to pick the wall permittivity.
FREQ_GHZ = 2.4
WAVELENGTH_M = 0.299792458 / FREQ_GHZ  # c = 299 792 458 m/s (SI, exact)

# Concrete, ITU-R P.2040-1 (2015), Table 3: eps' = a f^b, sigma = c f^d with
# f in GHz, a = 5.24, b = 0, c = 0.0462, d = 0.7822. With exp(-i w t),
# loss is a POSITIVE imaginary part: eps = eps' + i sigma / (w eps0).
EPS0 = 8.8541878128e-12  # F/m, CODATA 2018
_SIGMA_CONCRETE = 0.0462 * FREQ_GHZ**0.7822  # S/m
EPS_CONCRETE = complex(5.24, _SIGMA_CONCRETE / (2.0 * np.pi * FREQ_GHZ * 1e9 * EPS0))

SOURCE_SIGMA = 0.2  # lambda
AVG_RADIUS = 1.0  # lambda; the receiver's local-mean window
PML_WIDTH = 2.0  # lambda
PML_STRENGTH = 3.0  # a in s = 1 + i a (d/L)^2; normal-incidence R ~ exp(-2kaL/3)

# The interior of the scene, in wavelengths.
WIDTH = 24.0
HEIGHT = 18.0
TX_TRUE = (5.3, 9.1)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned block of material: x0 <= x <= x1, y0 <= y <= y1."""

    x0: float
    x1: float
    y0: float
    y1: float
    eps: complex = EPS_CONCRETE

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return (x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1)


@dataclass(frozen=True)
class Scene:
    name: str
    blocks: tuple[Box, ...] = ()
    tx: tuple[float, float] = TX_TRUE
    width: float = WIDTH
    height: float = HEIGHT

    def eps(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Relative permittivity n^2 at points; air is 1."""
        e = np.ones(np.broadcast(x, y).shape, complex)
        for b in self.blocks:
            e[b.contains(x, y)] = b.eps
        return e

    def in_material(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = np.zeros(np.broadcast(x, y).shape, bool)
        for b in self.blocks:
            m |= b.contains(x, y)
        return m


def free_space() -> Scene:
    return Scene("free")


def floor_plan() -> Scene:
    """Three rooms in a row, the right one split in two, and a pillar.

    Walls are one wavelength (12.5 cm at 2.4 GHz) of concrete, each with a
    doorway. The transmitter sits in the left room, so the far rooms are
    reached only through walls and doors -- which the log-distance law does
    not know about.
    """
    t = 1.0
    blocks = (
        # wall 1 at x = 9..10, door at y = 12..14
        Box(9.0, 9.0 + t, 0.0, 12.0),
        Box(9.0, 9.0 + t, 14.0, HEIGHT),
        # wall 2 at x = 16..17, door at y = 3..5
        Box(16.0, 16.0 + t, 0.0, 3.0),
        Box(16.0, 16.0 + t, 5.0, HEIGHT),
        # right-hand rooms split at y = 9..10, door at x = 20..21.5
        Box(17.0, 20.0, 9.0, 9.0 + t),
        Box(21.5, WIDTH, 9.0, 9.0 + t),
        # a square pillar in the middle room
        Box(12.5, 13.5, 5.5, 6.5),
    )
    return Scene("walls", blocks)


SCENES = {"free": free_space, "walls": floor_plan}


# ---------------------------------------------------------------------------
# the solver
# ---------------------------------------------------------------------------


@dataclass
class Field:
    """A solved scene on its grid. Arrays are indexed [j, i] = [y, x]."""

    scene: Scene
    ppw: int
    x: np.ndarray  # (nx,) node coordinates, lambda
    y: np.ndarray  # (ny,)
    u: np.ndarray  # (ny, nx) complex field
    eps: np.ndarray  # (ny, nx) permittivity used (sub-cell averaged)
    seconds: float = 0.0
    meta: dict = field(default_factory=dict)

    @property
    def h(self) -> float:
        return 1.0 / self.ppw

    def power_db(self) -> np.ndarray:
        """Raw received power 10 log10 |u|^2, with the small-scale fading."""
        return 10.0 * np.log10(np.abs(self.u) ** 2 + 1e-300)

    def local_mean_db(self, radius: float = AVG_RADIUS) -> np.ndarray:
        """10 log10 of |u|^2 averaged over a disk: what a receiver reads."""
        r = int(np.ceil(radius * self.ppw))
        g = np.arange(-r, r + 1) * self.h
        kern = (g[None, :] ** 2 + g[:, None] ** 2 <= radius**2).astype(float)
        kern /= kern.sum()
        pw = fftconvolve(np.abs(self.u) ** 2, kern, mode="same")
        return 10.0 * np.log10(np.maximum(pw, 1e-300))

    def interior_mask(self, margin: float = AVG_RADIUS) -> np.ndarray:
        """Nodes whose averaging disk lies inside the physical domain."""
        gx, gy = np.meshgrid(self.x, self.y)
        w, h = self.scene.width, self.scene.height
        return (np.abs(gx - w / 2) <= w / 2 - margin) & (
            np.abs(gy - h / 2) <= h / 2 - margin
        )

    def at(self, arr: np.ndarray, xq: np.ndarray, yq: np.ndarray) -> np.ndarray:
        """Nearest-node lookup. Probes are placed on nodes, so this is exact."""
        i = np.rint((np.asarray(xq) - self.x[0]) / self.h).astype(int)
        j = np.rint((np.asarray(yq) - self.y[0]) / self.h).astype(int)
        return arr[j, i]


def _stretch(coord: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """PML factor s = 1 + i a (d/L)^2, d the depth into the layer."""
    d = np.maximum(lo - coord, 0.0) + np.maximum(coord - hi, 0.0)
    return 1.0 + 1j * PML_STRENGTH * (d / PML_WIDTH) ** 2


def _second_diff(n: int, h: float, s_half: np.ndarray) -> sp.csr_matrix:
    """d/dx (1/s du/dx) with u = 0 beyond both ends; s at the n+1 half nodes."""
    g = sp.diags([-np.ones(n), np.ones(n)], [-1, 0], shape=(n + 1, n)) / h
    return (-(g.T @ sp.diags(1.0 / s_half) @ g)).tocsr()


def _eps_subcell(scene: Scene, X: np.ndarray, Y: np.ndarray, h: float, m: int = 4):
    """Permittivity averaged over m x m sub-samples of each cell.

    Plain point sampling moves an interface by up to h/2 between grids and
    makes the convergence study measure the staircase; averaging makes the
    wall's thickness the same on every grid to O(h^2).
    """
    offs = (np.arange(m) + 0.5) / m - 0.5
    acc = np.zeros(X.shape, complex)
    for dx in offs:
        for dy in offs:
            acc += scene.eps(X + dx * h, Y + dy * h)
    return acc / m**2


def solve(scene: Scene, ppw: int = 16) -> Field:
    """Solve the Helmholtz equation for `scene` at `ppw` points per wavelength."""
    require_range(ppw, 4, 64, "points per wavelength")
    t0 = time.time()
    h = 1.0 / ppw
    x = np.arange(
        -PML_WIDTH, scene.width + PML_WIDTH + 0.5 * h, h
    )  # nodes at multiples of h, so integer-lambda probes are nodes
    y = np.arange(-PML_WIDTH, scene.height + PML_WIDTH + 0.5 * h, h)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y)

    sx = _stretch(x, 0.0, scene.width)
    sy = _stretch(y, 0.0, scene.height)
    sx_half = _stretch(np.concatenate([[x[0] - h / 2], x + h / 2]), 0.0, scene.width)
    sy_half = _stretch(np.concatenate([[y[0] - h / 2], y + h / 2]), 0.0, scene.height)

    eps = _eps_subcell(scene, X, Y, h)
    dxx = _second_diff(nx, h, sx_half)
    dyy = _second_diff(ny, h, sy_half)
    SX, SY = np.meshgrid(sx, sy)
    A = (
        sp.kron(sp.diags(sy), dxx)
        + sp.kron(dyy, sp.diags(sx))
        + sp.diags((K0**2 * eps * SX * SY).ravel())
    ).tocsc()

    xt, yt = scene.tx
    f = np.exp(-((X - xt) ** 2 + (Y - yt) ** 2) / (2 * SOURCE_SIGMA**2))
    f /= f.sum() * h * h  # unit integral on THIS grid
    rhs = -(SX * SY * f).ravel()

    u = splu(A, permc_spec="COLAMD").solve(rhs.astype(complex))
    u = u.reshape(ny, nx)
    require(bool(np.all(np.isfinite(u))), "Helmholtz solve returned non-finite values")
    return Field(
        scene=scene,
        ppw=ppw,
        x=x,
        y=y,
        u=u,
        eps=eps,
        seconds=time.time() - t0,
        meta={"n_unknowns": nx * ny, "h": h},
    )


# ---------------------------------------------------------------------------
# the analytic answer in free space
# ---------------------------------------------------------------------------


def analytic_free_space(r: np.ndarray) -> np.ndarray:
    """(i/4) H0^(1)(k r) times the Gaussian source's form factor."""
    return (
        0.25j
        * hankel1(0, K0 * np.asarray(r, float))
        * np.exp(-0.5 * (K0 * SOURCE_SIGMA) ** 2)
    )


def free_space_law_db(d: np.ndarray) -> np.ndarray:
    """Large-r limit of 10 log10 |u|^2:  |H0(kr)|^2 -> 2 / (pi k r).

    This is the log-distance law with n = 1 and d0 = 1 lambda, and its P0 is
    the oracle's published constant.
    """
    d = np.asarray(d, float)
    return 10.0 * np.log10(np.exp(-((K0 * SOURCE_SIGMA) ** 2)) / (8.0 * np.pi * K0 * d))


P0_FREE_SPACE = float(free_space_law_db(np.array(1.0)))  # dB at d0 = 1 lambda
N_FREE_SPACE = 1.0  # cylindrical spreading in 2-D


def analytic_check(ppw: int = 16, r_min: float = 2.0) -> dict:
    """Numerical free-space field against the Hankel function.

    Tests the discretisation and the PML together: a reflecting boundary
    would show up as a standing-wave error that grows toward the edges.
    """
    fld = solve(free_space(), ppw)
    X, Y = np.meshgrid(fld.x, fld.y)
    r = np.hypot(X - fld.scene.tx[0], Y - fld.scene.tx[1])
    m = fld.interior_mask(margin=0.0) & (r >= r_min)
    exact = analytic_free_space(r[m])
    num = fld.u[m]
    rel = np.abs(num - exact) / np.abs(exact)
    db_err = 10.0 * np.log10(np.abs(num) ** 2 / np.abs(exact) ** 2)
    return {
        "ppw": ppw,
        "r_min": r_min,
        "median_rel_error": float(np.median(rel)),
        "p95_rel_error": float(np.quantile(rel, 0.95)),
        "max_abs_db_error": float(np.max(np.abs(db_err))),
        "rms_db_error": float(np.sqrt(np.mean(db_err**2))),
        "seconds": fld.seconds,
    }


# ---------------------------------------------------------------------------
# grid convergence (rule 4)
# ---------------------------------------------------------------------------

# Probes on integer-lambda nodes, which are nodes on every grid with an
# integer ppw. Spread over the three rooms, behind walls and in doorways.
PROBES = (
    (3.0, 4.0),
    (7.0, 15.0),
    (12.0, 9.0),
    (13.0, 13.0),
    (14.0, 3.0),
    (19.0, 4.0),
    (20.0, 14.0),
    (22.0, 7.0),
)


CONVERGENCE_PPWS = (8, 10, 12, 16, 20, 24, 32)


def grid_convergence(
    scene_name: str = "walls", ppws=CONVERGENCE_PPWS, step: float = 0.5
) -> list[dict]:
    """How much does the measured field move as the grid is refined?

    Reported against the finest grid, at the probes and over every node of a
    common `step`-lambda lattice inside the averaging margin: the local-mean
    power (the arms' target) and the raw power (with fading, which is far
    more sensitive to the phase error of the scheme).
    """
    scene = SCENES[scene_name]()
    fields = []
    for p in ppws:
        fields.append(solve(scene, p))
        print(f"   ppw {p}: {fields[-1].seconds:.0f} s", flush=True)
    ref = fields[-1]
    xs = np.arange(AVG_RADIUS, scene.width - AVG_RADIUS + 1e-9, step)
    ys = np.arange(AVG_RADIUS, scene.height - AVG_RADIUS + 1e-9, step)
    LX, LY = np.meshgrid(xs, ys)
    keep = ~scene.in_material(LX, LY)
    LX, LY = LX[keep], LY[keep]
    px = np.array([p[0] for p in PROBES])
    py = np.array([p[1] for p in PROBES])

    def views(fl):
        lm, raw = fl.local_mean_db(), fl.power_db()
        return fl.at(lm, LX, LY), fl.at(raw, LX, LY), fl.at(lm, px, py)

    ref_lm, ref_raw, ref_probe = views(ref)
    rows = []
    for fl in fields:
        lm, raw, probe = views(fl)
        d_lm = np.abs(lm - ref_lm)
        row = {
            "scene": scene_name,
            "ppw": fl.ppw,
            "h": fl.h,
            "n_unknowns": fl.meta["n_unknowns"],
            "seconds": fl.seconds,
            "local_mean_max_db": float(d_lm.max()),
            "local_mean_rms_db": float(np.sqrt(np.mean(d_lm**2))),
            "local_mean_p99_db": float(np.quantile(d_lm, 0.99)),
            "raw_rms_db": float(np.sqrt(np.mean((raw - ref_raw) ** 2))),
            "raw_p95_db": float(np.quantile(np.abs(raw - ref_raw), 0.95)),
            "probe_max_db": float(np.max(np.abs(probe - ref_probe))),
        }
        for k, v in enumerate(probe):
            row[f"probe{k}_db"] = float(v)
        rows.append(row)
    return rows


def choose_ppw(rows: list[dict], tol_db: float = 0.5) -> int:
    """Coarsest grid whose local-mean map is within `tol_db` of the finest
    at 99% of the lattice points.

    The 99th percentile rather than the maximum: in the deepest interference
    minima a local mean of -70 dB moves by several dB for a change in |u| that
    no receiver could see through 2 dB of noise, and the maximum would be
    set by those few points alone. The finest grid itself is excluded: its
    zero difference against itself says nothing. The tolerance is a quarter
    of the 2 dB receiver noise.
    """
    ok = [r["ppw"] for r in rows[:-1] if r["local_mean_p99_db"] <= tol_db]
    return int(min(ok)) if ok else int(rows[-1]["ppw"])
