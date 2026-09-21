# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-21

First public release.

### Added

- **Data layer** (`physprior.data`) — one loader per source (GWOSC, COBE/FIRAS,
  NIST ASD, JPL Horizons), each downloading once, recording provenance
  (URL, byte count, SHA-256) and asserting the units it promises.
- **Physics problems** (`physprior.problems`) — `gravity`, `relativity` and
  `quantum`, each holding its simulations, its real-data benchmark tracks and
  its law-discovery experiments side by side.
- **Method arms** — `oracle`, `physics`, `pinn` (neural correction or ODE
  residual), `sr` (PySR), `nn` (tuned MLP baseline).
- **Benchmark protocol** (`physprior.benchmark`) — interpolation, data
  efficiency, noise, extrapolation, parameter recovery and law recovery, run
  identically for every problem.
- **Simulations** — symplectic and RK4 integrators with conservation
  diagnostics; the three-body figure-eight and a measured Lyapunov exponent;
  Schwarzschild orbits and photon geodesics; a Schrödinger eigenvalue solver
  with Richardson extrapolation; split-operator tunnelling.
- **Animations** (`physprior.viz.animate`) — orbits, wavefunctions, and the
  training process with the trainable constant plotted per epoch.
- **CLI** — `physprior run | report | figures | notebooks | data | info`.

### Notable results

- GW150914's detector-frame chirp mass recovered from raw LIGO strain, with a
  post-Newtonian order ablation showing the Newtonian law biased by ~+9 M☉.
- General relativity detected in Mercury's DE441 acceleration,
  α = 1.000121 ± 0.000017, and independently by integrating the Schwarzschild
  orbit equation (42.98 arcsec/century from both).
- The Rydberg formula rediscovered by symbolic regression from NIST levels,
  and the 10.8 ppm relativistic + QED shift detected two independent ways.
- Symbolic regression **fails** to recover Planck's law from the FIRAS band,
  with a controlled band-widening experiment isolating the cause.

### Fixed during development

- The GW frequency extraction's SNR threshold was calibrated on injected
  signals rather than chosen a priori; the original value biased a known
  chirp mass by −70%.
- Perihelion location changed from grid parabola-fitting to root-finding on
  the dense solution, which the 5e-7 rad per orbit effect requires.
- The noise sweep now scores against clean held-out values, not noisy ones.

[Unreleased]: https://github.com/drorjacoby/physics-prior/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/drorjacoby/physics-prior/releases/tag/v0.1.0
