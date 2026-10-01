# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **The Lorenz study** (`physprior lorenz`, `physprior.lorenz`). An inverse
  problem on a chaotic system: 40 noisy observations of one Lorenz-63
  trajectory; reconstruct it, recover (sigma, rho, beta) from a start wrong by
  a factor of two, and forecast. Arms: a tuned black box (and its derivative
  regressed on the law), finite differences, single shooting, multiple
  shooting, the PINN, and the PINN polished by shooting. Sweeps over noise
  (0 to 20 %) and budget (10 to 160 points); an optimisation ladder that adds
  the training recipe one piece at a time, plus gradient-norm balancing and
  causal weights; the cost of forward-mode, `torch.func.jvp` and reverse-mode
  derivatives, with and without `torch.compile`; the Lyapunov exponent from
  24 trajectory pairs and the forecast horizon against the error in the
  constants. Page generated from results: `docs/lorenz/README.md`.
- **T14, the butterfly and the PINN**: the study as an executable tutorial,
  with the block diagram, the loss term by term, and a vanilla PINN failing
  beside the recipe that fixes it.
- **`docs/SUMMARY.md`** (`physprior summary-md`): the project on one page,
  generated from `results/`, with the pipeline and the Lorenz PINN as
  diagrams. The summary notebook opens with the Lorenz study.

- **H7: the physics weight against the amount of training data**
  (`physprior budget-weight`). The `pinn` arm's weight is chosen per budget
  of the data-budget sweep on the tuning seeds and re-scored on the
  reporting seeds. Predictions committed first; supported, narrowly: the
  weight for few points is at least the weight for many on 4 of 5 tracks
  (not on the CMB, where the law is already right), and the per-budget weight
  is no worse than a single weight at 17 of 25 cells.
- **`docs/MISSIONS.md`**: every task with its goal, its conclusion for the
  PINN and the physics prior, and its status; clear results first, open ones
  after them with what would move each.
- **`notebooks/summary.ipynb`** (`physprior summary --execute`): the method,
  a task chooser, and each result mission as its data and results in simple
  plots, computed from `results/` at run time.
- **`gravity/pulsar_spindown`**: the braking index `n = ν ν̈ / ν̇²` of the
  ATNF Pulsar Catalogue's young isolated pulsars (v2.8.1, 13 of 4393 by a
  rule fixed before any fit), against the vacuum-dipole value 3, with a
  simulated control where the index depends on age through a known term. The
  four predictions were committed before the run and all hold: the dipole
  law is incomplete, every arm fails on the older glitch-dominated pulsars,
  the tuning rule prefers no correction on the real data, and the correction
  does learn the age dependence in the control. New loader
  `data/sources/atnf.py`; new sourced constants `[DIPOLE]`, `[MT77]`.
- **`quantum/helium`**: the 452 singly excited He I terms from the NIST ASD,
  with the hydrogenic law, a Rydberg–Ritz comparison fitted on the same
  split, and a quantum-defect diagnostic. It is the H3 test named in advance,
  and H3's own refutation criterion is met: the frozen `pinn` arm reduces to
  the `physics` fit, and neither it nor an unbalanced variant learns the
  defect's l structure. Symbolic regression beats the hydrogenic law out of
  range.
- **`fields`**, a new problem. `fields/weather`: NOAA ISD monthly-mean station
  temperatures over the Alps, the lapse rate as the physics prior,
  extrapolation up the mountain, kriging on its own and on the law's
  residuals, and a simulated control that checks the error bars.
  `fields/rf`: a 2-D Helmholtz radio field with a grid-convergence study, the
  log-distance law with the transmitter position fitted, free space against
  walls. New extra: `fields` (scikit-learn, for the GP arms).
- **`physprior reconstruct`**: sparse-sensor field reconstruction in 1-D, 2-D
  and 3-D, with the hypothesis written before the runs; the physics prior's
  advantage grows with dimension on both field families.
- **`physprior dynamics`**: learned time-steppers for ODEs and PDEs, compared
  over long rollouts against nine expectations written in advance.
- **`physprior optim`**: a NumPy network with hand-written backprop and
  optimizers; optimizer, learning-rate, loss-function and curvature studies
  for physics fits, PINNs and black boxes; and a measurement of what loss
  balancing does to the `pinn` arm (see Changed).
- **`physprior theory`**: in-repo symbolic regression (exhaustive and genetic
  search, PySR's selection rule, SINDy), measured against PySR, with
  `docs/theory/` on how symbolic regression and the fitting packages work.
- Tutorials **T8–T13** and the problem notebook **`04_fields`**.

- **The reproduction contract.**
  - `physprior verify CANDIDATE` compares a regenerated results directory with
    the committed one, number by number (relative tolerance 1e-9). It
    ignores only wall-clock `seconds` and `environment.json`, and exits 1 on
    any other difference.
  - `make reproduce` re-runs the pipeline into `build/reproduce/`, touching
    nothing committed, then verifies.
  - `physprior report --check` fails if the README or `docs/RESULTS.md`
    tables no longer match `results/`. The test suite runs it, so CI does too.
  - `physprior run` and `physprior neglected` write `results/environment.json`:
    git commit and dirty flag, Python, platform, package versions, torch
    device, and the `requirements.lock` hash.
- **`docs/RELATED_WORK.md`** and a checked **`docs/references.bib`**: PINNs,
  failure modes, loss balancing, APHYNITY and model discrepancy, PySR and AI
  Feynman, the benchmark suites, and weak baselines, each with what this
  project does that the cited work does not.
- **Seed spread in the extrapolation tables.** `physprior report` now gives
  `n_seeds` and the per-seed `out/in` range beside each median. A new table,
  *Against the fitted law, seed by seed*, pairs each arm with `physics` on
  the same seed. It states a verdict only when every reporting seed agrees,
  and marks the result `mixed` otherwise. The PINN-vs-physics out-of-range
  result is `mixed` on `gravity/kepler` and `quantum/cmb`.
  `tests/test_claims.py` now requires the >100× black-box gap on every seed,
  not only in the median.
- **`docs/HYPOTHESES.md`**: the six physical questions the project answers,
  the evidence for each, and what would refute it. H3 (a correction helps
  when the missing physics is distinguishable from the law) is the open one,
  with predictions written down before the helium track exists.
- **`docs/DECISIONS.md`**: the research decisions taken, and the open ones
  gathered from the phase plans.
- **Mercury: the converged residual explained.** The remaining α − 1 is
  not numerical (Richardson). It is two omissions of opposite sign, each
  about 4×10⁻⁴, which nearly cancel: the Sun's oblateness and frame dragging,
  and the Sun's barycentric motion in the n-body relativistic (EIH)
  equations. With DE441's own model, α = 1 within its error.
  `neglected_terms()`, `_eih_1pn()` and `richardson_truncation()` in
  `problems/relativity/mercury.py`; `horizons.vectors(..., centre="ssb")`;
  new sourced constants `[DE440]`; `results/relativity/mercury/
  neglected_terms.csv`; `tests/test_mercury_model.py` checks EIH against its
  Schwarzschild limit.

### Changed

- **The `pinn` arm no longer uses loss balancing; each track has its own
  physics weight, tuned.** Balancing raised the weight without bound, because
  this arm's penalty mean(NN²) has a gradient that vanishes with the
  correction, and switched the correction off. The weight is now a
  per-track hyperparameter chosen on the tuning seeds from a validation block
  at the top of the training range (`benchmark/pinn_tuning.py`, and
  `problems/fields/retune.py` for the fields tracks), with ties to the larger
  weight and grid edges extended. The `pinn` rows of every committed sweep
  were recomputed with the settings each sweep was run with; no other arm's
  rows changed. With the correction back on, the arm extrapolates worse
  than the fitted law on hydrogen, CMB and helium on every reporting seed,
  and on Kepler in the median; where the validation block prefers no
  correction (weather in July, both radio scenes) the weight is pinned high
  and the arm is the law by choice. Helium's `pinn_unbalanced` diagnostic is
  replaced by `pinn_balanced`, the old configuration. `optim/pinn_balance.py`
  now names `balance=True` explicitly instead of the frozen default.
- **The reading of the frozen `pinn` arm.** Loss balancing drives the physics
  weight up without bound and switches the learned correction off, so on the
  algebraic tracks the balanced `pinn` is the `physics` fit plus a vanishing
  correction. Phase 2's gain was measured against an unbalanced PINN whose
  correction overfits. README, `docs/plans/PLAN.md` §5A and
  `docs/DECISIONS.md` said so, and the configuration was then changed
  (above).
- The notebook helpers moved to `reporting/cells.py`, so topic modules can
  share them without a circular import.
- `docs/TOOLING.md` versions now match `requirements.lock`.

## [0.2.0] - 2026-09-25

### Added

- **`physprior neglected`**: the neglected-terms study is now reproducible by
  a command. Its five tables and fourteen figures were committed with nothing
  in the repository that regenerated them; `scripts/regenerate.sh` skipped the
  study entirely. Stages: `algebraic`, `ode`, `pde`, `detail`, `derivative`,
  `tune`. A test asserts that no committed figure is left without a stage.
- **`docs/neglected/`**: a topic page for the study, which had none. Every
  number in it is re-derived from `results/` by `tests/test_claims.py`.
- **`physprior.methods.device`**: selectable device and dtype via
  `PHYSPRIOR_DEVICE` / `PHYSPRIOR_DTYPE`, reported by `physprior info`.
  `resolve()` returns the pair because MPS has no float64 kernels, so
  choosing that device chooses single precision.
- **`benchmarks/`**: what a GPU is worth here, measured rather than assumed
  (~7× on the 2-D field residual at 10⁴ collocation points), plus why
  `mps.is_available()` is False on a machine that has an M2: torch 2.11
  requires macOS 14 and this machine runs 13.4.
- **`config.short_path`**: printing an output path no longer crashes when
  the output directory is redirected outside the project root.
- Residual-adaptive collocation, a curriculum in `t`, and a curvature
  penalty (`w_smooth`) on the PDE residual PINN. None of them fixes it; all
  are recorded because each is a standard remedy that did not work here.

### Fixed

- `Path.relative_to(root)` raised whenever `PHYSPRIOR_RESULTS_DIR` or
  `PHYSPRIOR_FIGURES_DIR` pointed outside the checkout, after the work was
  already done. Affected `reporting/figures.py` as well.
- Three mypy errors introduced by the new modules.

### Changed

- The project brief's "Current state" section claimed two commits at twenty-four, and
  mentioned neither the arm that does not converge nor the GPU situation.
- `methods/device.py` previously argued that a float32 second derivative
  "loses about half its digits" and that the GPU was therefore unsafe for a
  recovered constant. Measured on the analytic field, the cost is 2×10⁻⁸
  relative, while the network's own error in `u_xx` is 60 %. The digit count
  was right and the conclusion was wrong.

### Investigated and ruled out

- The free correction `C(u, u_x)` was the obvious explanation for the PDE
  arm's unidentifiable `alpha`, since `(alpha, C)` admits any `alpha`.
  Tested at `eps = 0`, where the true correction is exactly zero: a 10⁶
  range on `w_phys` and then removing `C` outright move `alpha` by 0.0013,
  in the wrong direction. Refuted. `results/neglected/tune_pde_correction.csv`.

### Known not to converge

- The 2-D residual PINN on the PDE rung returns `alpha` about 69 % wrong at
  `eps = 0`, where the modelled law is exactly right. Marked
  `converged=False`; the rung's conclusion rests on the `physics` arm, whose
  result is a closed-form identity. The diagnosis (a data loss of 8×10⁻⁴
  with `u_xx` 50–60 % too large, so `alpha = |u_t|/|u_xx|` cannot be right)
  is in `docs/METHOD.md`.

## [0.1.0] - 2026-09-21

First public release.

### Added

- **Data layer** (`physprior.data`): one loader per source (GWOSC, COBE/FIRAS,
  NIST ASD, JPL Horizons), each downloading once, recording provenance
  (URL, byte count, SHA-256) and asserting the units it promises.
- **Physics problems** (`physprior.problems`): `gravity`, `relativity` and
  `quantum`, each holding its simulations, its real-data benchmark tracks and
  its law-discovery experiments side by side.
- **Method arms**: `oracle`, `physics`, `pinn` (neural correction or ODE
  residual), `sr` (PySR), `nn` (tuned MLP baseline).
- **Benchmark protocol** (`physprior.benchmark`): interpolation, data
  efficiency, noise, extrapolation, parameter recovery and law recovery, run
  identically for every problem.
- **Simulations**: symplectic and RK4 integrators with conservation
  diagnostics; the three-body figure-eight and a measured Lyapunov exponent;
  Schwarzschild orbits and photon geodesics; a Schrödinger eigenvalue solver
  with Richardson extrapolation; split-operator tunnelling.
- **Animations** (`physprior.viz.animate`): orbits, wavefunctions, and the
  training process with the trainable constant plotted per epoch.
- **CLI**: `physprior run | report | figures | notebooks | data | info`.

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

[Unreleased]: https://github.com/drorjac/physics-prior/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/drorjac/physics-prior/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/drorjac/physics-prior/releases/tag/v0.1.0
