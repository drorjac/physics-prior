# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`physprior neglected`** — the neglected-terms study is now reproducible by
  a command. Its five tables and fourteen figures were committed with nothing
  in the repository that regenerated them; `scripts/regenerate.sh` skipped the
  study entirely. Stages: `algebraic`, `ode`, `pde`, `detail`, `derivative`,
  `tune`. A test asserts that no committed figure is left without a stage.
- **`docs/neglected/`** — a topic page for the study, which had none. Every
  number in it is re-derived from `results/` by `tests/test_claims.py`.
- **`physprior.methods.device`** — selectable device and dtype via
  `PHYSPRIOR_DEVICE` / `PHYSPRIOR_DTYPE`, reported by `physprior info`.
  `resolve()` returns the pair because MPS has no float64 kernels, so
  choosing that device chooses single precision.
- **`benchmarks/`** — what a GPU is worth here, measured rather than assumed
  (~7× on the 2-D field residual at 10⁴ collocation points), plus why
  `mps.is_available()` is False on a machine that has an M2: torch 2.11
  requires macOS 14 and this machine runs 13.4.
- **`config.short_path`** — printing an output path no longer crashes when
  the output directory is redirected outside the project root.
- Residual-adaptive collocation, a curriculum in `t`, and a curvature
  penalty (`w_smooth`) on the PDE residual PINN. None of them fixes it; all
  are recorded because each is a standard remedy that did not work here.

### Fixed

- `Path.relative_to(root)` raised whenever `PHYSPRIOR_RESULTS_DIR` or
  `PHYSPRIOR_FIGURES_DIR` pointed outside the checkout — after the work was
  already done. Affected `reporting/figures.py` as well.
- Three mypy errors introduced by the new modules.

### Changed

- the project brief's "Current state" claimed two commits at twenty-four, and
  mentioned neither the arm that does not converge nor the GPU situation.
- `methods/device.py` previously argued that a float32 second derivative
  "loses about half its digits" and that the GPU was therefore unsafe for a
  recovered constant. Measured on the analytic field, the cost is 2×10⁻⁸
  relative — while the network's own error in `u_xx` is 60 %. The digit count
  was right and the conclusion was wrong.

### Known not to converge

- The 2-D residual PINN on the PDE rung returns `alpha` about 69 % wrong at
  `eps = 0`, where the modelled law is exactly right. Marked
  `converged=False`; the rung's conclusion rests on the `physics` arm, whose
  result is a closed-form identity. The diagnosis — a data loss of 8×10⁻⁴
  with `u_xx` 50–60 % too large, so `alpha = |u_t|/|u_xx|` cannot be right —
  is in `docs/METHOD.md`.

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
