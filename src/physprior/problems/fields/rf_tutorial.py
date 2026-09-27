"""Notebook section for the radio-propagation track.

`rf_cells()` returns the cells; the tutorial module that owns the notebook
decides where they go. The cells solve a small scene live (a few seconds)
and read the full study's numbers from results/, so nothing is typed by hand.
"""

from __future__ import annotations

from physprior.reporting.cells import code, md


def rf_cells() -> list:
    return [
        md(r"""
## Radio propagation: where is the transmitter?

A transmitter at an unknown position radiates at a fixed frequency. A handful
of receivers report the power they see, in dB, with noise. Two questions:
**what is the power everywhere** (the coverage map), and **where is the
transmitter**?

### The forward model

A time-harmonic field $u(x, y)\,e^{-i\omega t}$ obeys the Helmholtz equation

$$\nabla^2 u + k^2 n(x, y)^2 u = -f(x, y),$$

with $k = 2\pi/\lambda$, $n$ the complex refractive index (1 in air; about
$2.3 + 0.15i$ for concrete at 2.4 GHz, the imaginary part being loss), and
$f$ the source. `rf_sim.solve` discretises it with second-order finite
differences and closes the domain with a perfectly matched layer, a strip in
which the coordinates are stretched into the complex plane so that outgoing
waves decay without reflecting. The sparse system is solved directly.

Lengths are in wavelengths. The source is a small Gaussian rather than a
point, because the field of a 2-D point source is singular at the source and
would never converge with the grid.
"""),
        code("""
import numpy as np, matplotlib.pyplot as plt
from physprior.problems.fields import rf, rf_sim as S

# a small room: 10 x 8 wavelengths, one wall with a doorway
room = S.Scene(
    "demo",
    blocks=(S.Box(5.0, 6.0, 0.0, 4.5), S.Box(5.0, 6.0, 6.0, 8.0)),
    tx=(2.3, 3.1),
    width=10.0,
    height=8.0,
)
fld = S.solve(room, ppw=10)
print(f"{fld.meta['n_unknowns']} unknowns, solved in {fld.seconds:.1f} s")

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, arr, title in [
    (axes[0], fld.power_db(), "raw power |u|^2 [dB]: multipath fading"),
    (axes[1], fld.local_mean_db(), "local-mean power [dB]: what a receiver reads"),
]:
    im = ax.imshow(arr, origin="lower", cmap="magma",
                   extent=(fld.x[0], fld.x[-1], fld.y[0], fld.y[-1]),
                   vmin=arr.max() - 45, vmax=arr.max())
    ax.set_title(title, loc="left"); ax.grid(False)
    rf._walls(ax, room)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.show()
"""),
        md(r"""
The raw power shows interference fringes half a wavelength apart: waves
reflected by the wall add to and cancel the direct wave. (The domain edges
reflect nothing: that is what the matched layer is for.) A received-signal
strength is an average over that fading, so the receivers here read the mean of
$|u|^2$ over a disk of one wavelength. The wall's shadow and the beam through
the doorway survive the averaging; the fringes do not.

### The law, and the inverse problem

The engineering model of path loss is the log-distance law

$$P(x, y) = P_0 - 10\, n \log_{10}\frac{d}{d_0}, \qquad
d = \lvert (x, y) - (x_t, y_t)\rvert .$$

In three dimensions free space has $n = 2$. In two dimensions a source
spreads over a circle rather than a sphere, $|u|^2 \sim 1/r$, so free space
has $n = 1$ here. Fitting the law to the readings is an inverse problem with
four unknowns: $P_0$, $n$, and the transmitter position $(x_t, y_t)$. The
position enters non-linearly, so the fit is started from several points.
"""),
        code("""
rng = np.random.default_rng(0)
X, Y = np.meshgrid(fld.x, fld.y)
lm = fld.local_mean_db()
ok = ((X > 1) & (X < 9) & (Y > 1) & (Y < 7) & ~room.in_material(X, Y)
      & (np.hypot(X - room.tx[0], Y - room.tx[1]) > 1.5))
pick = rng.choice(np.flatnonzero(ok.ravel()), 60, replace=False)
xr = np.column_stack([X.ravel()[pick], Y.ravel()[pick]])
yr = lm.ravel()[pick] + rng.normal(0, rf.NOISE_DB, len(pick))

fit = rf.fit_law(room, xr, yr)
p = fit.params
print(f"true transmitter      ({room.tx[0]:.2f}, {room.tx[1]:.2f})")
print(f"recovered             ({p['xt']:.2f}, {p['yt']:.2f})   "
      f"error {np.hypot(p['xt'] - room.tx[0], p['yt'] - room.tx[1]):.2f} wavelengths")
print(f"path-loss exponent n  {p['n']:.2f} +/- {fit.param_sigma['n']:.2f}   (free space: 1)")
"""),
        md(r"""
With a wall in the way the fitted exponent tends to come out above 1: the law has no
term for a wall, so it steepens its distance dependence to absorb the loss.
This is the same mechanism as a constant absorbing a missing term elsewhere
in the project, and it is why the recovered exponent is not a property of the
building.

### The comparison

The full study (`rf.run()`) uses a 24 x 18 wavelength floor plan with three
rooms, and a free-space control with the same receivers. The arms:

| arm | what it knows |
|---|---|
| `oracle` | the free-space law at the true position, with the analytic $P_0$ and $n = 1$ |
| `physics` | the law, all four parameters fitted |
| `pinn` | the law plus a network correction (shape B), started from the physics fit |
| `pinn_free` | the same with the physics weight fixed at 1, no loss balancing |
| `nn` | a tuned MLP on $(x, y)$ |
| `gp` | ordinary kriging, a Gaussian process with a constant mean |
| `kriging` | the fitted law as the mean, plus a GP on its residuals |

The free-space control is the case where the law is complete; the floor plan
is the case where it misses the walls. Every arm is scored against the true
map, not only against held-out noisy readings.
"""),
        code("""
from physprior.config import get_settings
try:
    t = rf.summary_tables()
except FileNotFoundError:
    t = None
    print("No results yet: run  rf.run()  (about an hour) or  rf.run(quick=True).")
if t is not None:
    print("Map RMSE against the truth [dB], median over reporting seeds")
    display(t["budget"].round(2))
    print("Transmitter error [wavelengths] and fitted exponent")
    display(t["tx"].round(2))
"""),
        code("""
from IPython.display import Image, display
fig_dir = get_settings().figures("fields")
for name in ("rf_scene", "rf_budget", "rf_tx_recovery", "rf_reconstruction_walls",
             "rf_extrapolation", "rf_helmholtz_window"):
    path = fig_dir / f"{name}.png"
    if path.exists():
        display(Image(filename=str(path)))
"""),
        md(r"""
### Reading the results

Three things to look for in the tables above.

1. **Free space.** The law is exact there, so `physics` should sit near the
   oracle and recover the transmitter; a correction (`pinn`, `kriging`) has
   nothing to find and can only add variance.
2. **The floor plan.** The law cannot make a step at a wall. Whatever an arm
   gains over `physics` here is the wall losses it learned from the
   receivers, and the gain should be largest where the receivers are dense.
3. **Extrapolation.** Trained only in the transmitter's room, no arm has seen
   a wall's effect. The law extrapolates its free-space shape through the
   walls; the data-driven arms extrapolate whatever their prior is.

A Helmholtz-residual PINN (shape A) is compared separately on a window of the
scene. It is given the refractive-index map, which the other arms are not,
and still faces a problem the readings do not pin down: power has no phase,
and a window with no boundary condition admits many solutions of the
equation.
"""),
    ]
