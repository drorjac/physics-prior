# Phase 5 - EM propagation, atmosphere, and loss discovery

The Phase 5 specification, as written. Every invariant in `CONTRIBUTING.md`
applies; where anything below conflicts with one, the invariant wins.

## 0. Precondition

Do not start until Phase 2 is merged and the PINN default config is frozen on
tune seeds 3/7/19. Every Phase 5 number is read against that frozen config.
If Phase 2 is still open, write docs/PHASE5_PLAN.md only and stop.

## 1. What this phase adds and why

| part | question it answers | why it belongs in physprior |
|---|---|---|
| A. loss discovery | Can symbolic regression find the right *loss*, not just the right law? | The loss is a prior on the noise, just as the law is a prior on the signal. Nothing in the repo tests the noise prior yet. |
| B1. `em/rain_attenuation` | Does a physics prior help when the law is itself an approximation (power law fitted to Mie scattering)? | Real EM propagation data, and a law that is known to be empirical. |
| B2. `atmos/barometric` | Can the arms recover a scale height and a lapse rate, and do they notice where the law breaks (inversions, tropopause)? | Real meteorological data with a known, localised law failure. |
| B3. `em/maxwell_1d` | Can a residual PINN solve Maxwell's equations and recover a permittivity, against an FDTD ground truth? | Simulation only. It stresses the PINN on wave problems, a known failure mode. |

Order: A (synthetic half) first, then B1, then A applied to B1, then B2, then B3.

## 2. Part A - Loss discovery

### 2.1 Hypothesis (write it in docs/METHOD.md before running anything)

For i.i.d. noise with density p, the maximum-likelihood loss l(r) = -log p(r)
is asymptotically efficient. So a correct loss search should rediscover -log p
up to an affine transform. If it finds something else, either the search is
broken or the noise is not what we injected. Both outcomes are reportable.

The analogue of the oracle arm is the **oracle loss** -log p_true. It is a
ceiling. Beating it out of sample needs an explanation, same rule as the oracle arm.

### 2.2 Setup

- Residual r = (y - f(x; theta)) / sigma.
- Inner problem: the physics arm as an M-estimator,
  theta_hat(l) = argmin sum_i l(r_i). Use `scipy.optimize.least_squares` with a
  callable `loss` (it takes rho(z) with z = r^2 and returns rho, rho', rho'').
- Loss family for the search: a monotone spline in |r| with l(0) = 0, symmetric,
  non-decreasing. Parameterise by knot values with a positivity transform.
- Outer objective: parameter-recovery error |theta_hat - theta_true| / sigma_theta,
  averaged over injections. It is **not** predictive error, because the question
  is identifiability.
- Outer optimiser: `scipy.optimize.differential_evolution` over knot values
  (no new dependency). An implicit-function-theorem gradient is optional and
  only worth doing if the black-box search is too slow.
- Stage 2: distil the learned spline into a formula with PySR over
  {+, -, *, /, abs, log, sqrt, square}. Cache under `.cache/sr/` as usual.

### 2.3 Controls (synthetic, on existing simulations)

| injected noise | expected l(r) up to affine | analytic check to put in tests |
|---|---|---|
| Gaussian | r^2 | median-vs-mean efficiency = 2/pi |
| Laplace | \|r\| | mean-vs-median efficiency = 1/2 |
| Student-t, nu in {2, 4} | log(1 + r^2/nu) | tail growth is logarithmic |
| Gaussian + 5 % outliers | Huber-like: quadratic core, sub-quadratic tail | no closed form, compare to Huber at its tuned delta |

Put the two efficiency values in `tests/` as analytic assertions computed in
code (1 / (4 n f(0)^2) vs variance / n), not typed constants.

### 2.4 Scoring - measured, never parsed

Same rule as law recovery. Do not compare PySR strings.
- Normalise both curves: subtract l(0), scale so they agree at |r| = 1.
- Report sup-norm distance to -log p on |r| in [0, 5].
- Report the **tail exponent** as the log-log slope of l over |r| in [2, 5],
  with a constancy check: 2 for Gaussian, 1 for Laplace, trending to 0 for Student-t.
- Report parameter-recovery error under: MSE, oracle loss, learned spline,
  distilled formula. The learned loss must land between MSE and the oracle loss,
  or the result needs an explanation.

### 2.5 Seeds

Noise realisations for the search: seeds 3/7/19. Reporting: 11/23/42.
The learned loss is frozen before any reporting seed is drawn.

### 2.6 Applying it to real data (after B1 exists)

On real data theta_true is unknown, so the outer objective cannot be computed
there. Do **not** optimise the loss on real data.
1. Fit the physics arm with MSE on the tune split of the real track.
2. Fit a noise model to those residuals (candidates: Gaussian, Student-t,
   Gaussian + outlier mixture, and skewed variants because wet-antenna
   attenuation is one-sided). Choose by held-out likelihood on the tune split.
3. Generate injections from the simulation half with that noise model.
4. Run the loss search on those injections. Freeze the result.
5. Apply the frozen loss to the reporting splits of the real track. Report the
   change in parameter estimates and their sigma vs MSE.

The deliverable is a statement of the form "the real residuals behave as if the
noise has tail exponent X", derived by the search, with its uncertainty.

## 3. Part B - New tracks

Each track needs: a loader under `data/sources/` with `require()` unit checks,
constants with sources in `constants.py`, a simulation half, `problem()`,
all six protocol questions, figures, and a notebook section with a generated
conclusion (Phase 1 machinery). Verify every download URL yourself, fetch only
through `data/cache.py`, record SHA-256, and put the source in the docstring.

### B1. `em/rain_attenuation`

| item | spec |
|---|---|
| real data | OpenMRG (Gothenburg CML + gauge dataset). Find the canonical archive and pin its checksum. |
| x | rain rate R from gauges, link frequency f, polarisation |
| y | specific attenuation gamma = A / L after baseline removal |
| law | gamma = k(f) R^alpha(f) |
| published theta | ITU-R P.838-3 k and alpha per frequency and polarisation. Store the table with its source. |
| simulation half | extinction from Mie scattering integrated over a drop-size distribution (Marshall-Palmer and a gamma DSD). Use `miepython` or an in-repo Mie series. Convergence study on the DSD quadrature and on the Mie series truncation. |
| what the simulation shows | the power law is only an approximation of the Mie integral. Measure how far it deviates as a function of f and R. That is the "law's own failure" control, like QED on hydrogen. |
| hard parts | wet-antenna offset, dry/wet classification, gauge-to-link spatial mismatch, DSD variability |
| signal-processing choices | baseline window and wet/dry threshold are calibrated on injections only (simulated rain on real dry periods). Never tuned on real rain events. |
| extrapolation splits | (i) train low R, test high R. (ii) train on a subset of frequencies, test on held-out frequencies. |
| parameter recovery | alpha(f) and k(f) vs ITU-R, with sigma |
| law recovery | exponent alpha measured from the log-log slope with a constancy check |

### B2. `atmos/barometric`

| item | spec |
|---|---|
| real data | IGRA v2 radiosonde soundings (NOAA NCEI). Pick stations and dates by a rule written down before looking at the profiles, for example a fixed station list and date range chosen from metadata only. |
| x, y | geopotential height z, pressure p (also T for a check) |
| law | hydrostatic + ideal gas with constant lapse rate: p = p0 (1 - Gamma z / T0)^(g / (R_d Gamma)) |
| theta | T0, Gamma (p0 from the surface level) |
| law failure | inversions and the tropopause, where Gamma changes. The PINN correction should localise there. Report where the correction net's magnitude peaks vs the measured tropopause height. |
| extrapolation | train on the lower troposphere, test up to and past the tropopause |
| constants | g, R_d with sources |

### B3. `em/maxwell_1d` (simulation only)

| item | spec |
|---|---|
| system | 1D Maxwell (E_z, H_y) through a dielectric slab of unknown eps_r, Gaussian pulse source |
| ground truth | Yee FDTD in `numerics/`, with a convergence study in dx and CFL number and Richardson extrapolation of a probe value |
| arms | residual PINN (Raissi-style, as for gw150914) solving the PDE and recovering eps_r from sparse probe measurements. Physics arm: FDTD forward model + `curve_fit` on eps_r. NN and SR on the probe time series. |
| what it tests | spectral bias on oscillatory fields, so Phase 2 Fourier features and causal weighting get a real test. Time extrapolation past the training window. |
| parameter recovery | eps_r vs injected value, sigma from ensembles vs curve_fit covariance |

### Stubs only (list in PHASE5_PLAN.md, do not implement)

- `em/gaseous_absorption`: oxygen absorption near 60 GHz, line-shape law, extrapolation across the line.
- `atmos/clausius_clapeyron`: saturation vapour pressure, where the constant-L assumption is the law's own failure.
- `em/two_ray`: two-ray ground-reflection path loss vs free space, with a break distance to recover.

## 4. Integration

- New problem families `em/` and `atmos/` follow the existing
  `<simulation>.py, <track>.py, discovery.py, run.py` layout.
- `physprior run em`, `physprior run atmos` and `run all` include them.
- New notebooks `04_em.ipynb`, `05_atmos.ipynb`, `06_loss_discovery.ipynb`,
  each ending in the generated verdict table, prose and falsification list.
- `results/headline.json` gets the new headline numbers. The README table
  regenerates through `physprior report`. Every narrative claim gets a
  `test_claims.py` assertion.
- docs/METHOD.md: the loss-discovery hypothesis (written before the run),
  and a post-mortem for anything that went wrong.

## 5. Stop points

1. After PHASE5_PLAN.md: wait for approval.
2. After Part A synthetic controls: report the table in 2.4. If Gaussian and
   Laplace controls fail, stop. The search is broken and nothing downstream
   can be trusted.
3. After B1 loader and simulation: report the power-law deviation from Mie
   before running any arm.

## 6. Final deliverable

One table, reporting seeds only:

| track or experiment | best arm (with TIE rule) | PINN vs physics on parameter recovery | law or loss recovered? | what failed |
|---|---|---|---|---|

Plus one paragraph per part stating the conclusion, generated from results.
Negative results at the same size as positive ones.