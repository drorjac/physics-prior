# The three physics problems

Each problem is a folder under `src/physprior/problems/` holding everything for that subject:
the simulations, the real-data tracks, the law-discovery experiments, and its
own results and figures.

```text
src/physprior/problems/<name>/
  <simulation>.py   orbits | spacetime | schrodinger -- where the law is
                    exactly what was put in, so a method's error is its own
  discovery.py      recover the law FROM the simulation, and calibrate the
                    pipeline that measures the real data
  <track>.py        a real-data benchmark track: problem() + run()
  run.py            runs all of the above -> results/ + figures/
results/<problem>/...       committed CSV + JSON
figures/<problem>/...       PNGs and GIFs
```

Run one with `physprior run gravity` (or `relativity`, `quantum`, `all`).

---

## gravity

**Simulations** (`problems/gravity/orbits.py`)

| What | Notes |
|---|---|
| `two_body` | Sun + one planet, started at perihelion from the vis-viva speed, in the centre-of-mass frame |
| `three_body` | the Chenciner–Montgomery figure-eight choreography, and a perturbed copy of it |
| `lyapunov_separation` | two figure-eights differing by 1 part in 10⁹, tracked until they separate — chaos as a measured exponent |
| `solar_system` | eight planets from **real JPL initial conditions**, integrated forward |
| `integrator_comparison` | velocity Verlet against RK4 over 200 years |

**Real data** — Kepler's third law from JPL DE441, eight planets.

**Discovery** (`problems/gravity/discovery.py`) — the force-law exponent and `GM`
from a simulated orbit; Kepler's 3/2 from simulated periods; and the same
force law recovered from the *chaotic* three-body run.

**Animations** — `two_body_mercury.gif`, `three_body_figure8.gif`,
`three_body_chaotic.gif`, `solar_system_inner.gif`, `pinn_learning_orbit.gif`.

---

## relativity

**Simulations** (`problems/relativity/spacetime.py`)

| What | Notes |
|---|---|
| `schwarzschild_orbit` | `d²u/dφ² + u = GM/h² + (3GM/c²)u²`, run with the GR term on and off |
| `mercury_precession` | the difference of those two runs, in arcsec/century |
| `inspiral_chirp` | a post-Newtonian binary inspiral with a damped ringdown |
| `light_deflection` | a photon's null geodesic past the Sun |

**Real data** — GW150914 strain from LIGO; Mercury's acceleration from DE441.

**Discovery** (`problems/relativity/discovery.py`)

- `injection_test` — a waveform with a **known** chirp mass injected into the
  **real detector noise** and pushed through the same extraction and fitting
  code, so the pipeline has an error budget.
- `threshold_calibration` — the extraction's SNR cut chosen on injections,
  never on the real event.
- `precession_recovery` — the GR coefficient from a simulated orbit, where the
  answer is exactly 1.

**Animations** — `schwarzschild_precession.gif`.

---

## quantum

**Simulations** (`problems/quantum/schrodinger.py`)

| What | Notes |
|---|---|
| `infinite_well`, `harmonic_oscillator`, `hydrogen_radial` | bound states by direct diagonalisation of a finite-difference Laplacian |
| `hydrogen_richardson` | two grids combined to cancel the leading error: 300 ppm → 4 ppm |
| `grid_convergence`, `box_convergence`, `r_min_sweep` | the solver's error, measured |
| `wavepacket` | split-operator tunnelling, unitary to 5×10⁻¹⁵ |

**Real data** — NIST hydrogen levels; the COBE/FIRAS blackbody.

**Discovery** (`problems/quantum/discovery.py`) — SR on the solved spectra
(`n²`, `(n+½)`, `−1/n²`); and `schrodinger_vs_nist`, which differences the
simulated non-relativistic atom against the real one to see QED.

**Animations** — `tunnelling.gif`.

---

## Why every problem has both

The simulations are the **control**. When the law is exactly what was put in,
whatever a method fails to recover is the method's own error. Three results in
this project exist only because both halves are present:

- the chirp-mass pipeline's SNR threshold was **wrong**, and only an injection
  with a known answer could show it;
- the +25% bias at Newtonian order is reproduced by the injection, so it is
  post-Newtonian truncation and not an artefact of the extraction;
- hydrogen's QED shift is reached twice — by fitting Bohr's law to NIST, and
  by solving Schrödinger's equation and differencing.
