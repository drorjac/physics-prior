# gravity

Newtonian orbital mechanics, where the law is exact and the question is what
a method can recover from data that obeys it.

**Real data** — Kepler's third law from the JPL DE441 ephemeris, eight
planets. **Simulations** — the control: when the law is exactly what was put
in, whatever a method fails to recover is the method's own error.

Code: [`src/physprior/problems/gravity`](../../src/physprior/problems/gravity)
· Results: [`results/gravity`](../../results/gravity)

---

## The simulations

| What | Notes |
|---|---|
| `two_body` | Sun + one planet, started at perihelion from the vis-viva speed, in the centre-of-mass frame |
| `three_body` | the Chenciner–Montgomery figure-eight choreography, and a perturbed copy of it |
| `lyapunov_separation` | two figure-eights differing by 1 part in 10⁹, tracked until they separate — chaos as a measured exponent |
| `solar_system` | eight planets from **real JPL initial conditions**, integrated forward |
| `integrator_comparison` | velocity Verlet against RK4 over 200 years |

<table>
<tr>
<td width="50%"><img src="../../figures/gravity/three_body_figure8.gif" alt="the figure-eight choreography"><br><sub><b>The figure-eight choreography.</b> Three equal masses on one closed orbit.</sub></td>
<td width="50%"><img src="../../figures/gravity/three_body_chaotic.gif" alt="a perturbed three-body orbit going chaotic"><br><sub><b>The same, perturbed.</b> One part in 10⁹ is enough.</sub></td>
</tr>
<tr>
<td><img src="../../figures/gravity/solar_system_inner.gif" alt="the inner solar system integrated from JPL initial conditions"><br><sub><b>The inner solar system</b>, from real JPL initial conditions.</sub></td>
<td><img src="../../figures/gravity/two_body_mercury.gif" alt="Sun and Mercury under velocity Verlet"><br><sub><b>Sun + Mercury.</b> <code>e = 0.206</code> makes perihelion the hard part.</sub></td>
</tr>
</table>

**The integrator is part of the physics model.** Velocity Verlet is
symplectic: its energy error oscillates and stays bounded. RK4 has the better
local error and is not symplectic, so its error *drifts*. At 4-day steps over
200 years Verlet stays at 2% while RK4 reaches 412% and the orbit unbinds.

![chaos measured as a Lyapunov exponent](../../figures/gravity/lyapunov.png)

## Recovering the law from the simulation

[`discovery.py`](../../src/physprior/problems/gravity/discovery.py) recovers
the force-law exponent and `GM` from a simulated orbit, Kepler's 3/2 from
simulated periods, and the same force law from the *chaotic* three-body run.

Symbolic regression returns the exponent as **−1.9999969** and `GM` to
**0.6 ppb** from a simulated orbit. That is the floor: when the same
machinery returns `GM_sun` from the real ephemeris with a tens-of-ppm
residual, the residual cannot be blamed on the method — it is the two-body
formula neglecting the planets' masses.

## The real-data track: `gravity/kepler`

Kepler's third law, `P = 2π√(a³/GM)`, fitted to eight planets.

| | |
|---|---|
| ![the track overview](../../figures/gravity/kepler/overview.png) | ![extrapolation beyond the fitted range](../../figures/gravity/kepler/extrapolation.png) |
| **Overview** — every arm on the track. | **Extrapolation** — trained on the inner planets, asked about the outer. |
| ![data efficiency](../../figures/gravity/kepler/data_efficiency.png) | ![the physics weight dial](../../figures/gravity/kepler/physics_weight.png) |
| **Data efficiency** — when does the black box catch up? | **The `w_phys` dial** — at `w_phys = 0` the recovered `GM_sun` is 19.5% wrong while the held-out error barely moves. |

A neural correction watching a constant converge:

<img src="../../figures/gravity/pinn_learning_orbit.gif" alt="a PINN learning an orbit while GM walks toward its published value" width="620">

## Caveats, stated up front

`P = 2π√(a³/GM)` ignores the planet's own mass and uses the osculating
semi-major axis. Both bias the recovered `GM_sun` at the tens-of-ppm level,
and the giant planets dominate the effect. Provenance for DE441 is in
[`docs/DATA.md`](../DATA.md).
