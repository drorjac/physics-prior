# Method — the decisions, and why

This file records the choices that would otherwise be invisible in the code,
including the ones that were made *after* something went wrong.

## Why these four datasets

Each was chosen so that a **published value exists to be checked against**,
and so that the law is genuinely non-trivial in at least one respect:

- **G (GW150914)** — the only track where the data is so scarce (7 points)
  that the black box cannot function at all, and the only one where the
  correct law is an *expansion* that can be truncated at the wrong order.
- **Q (FIRAS)** — a transcendental law, and a band that covers only part of
  its behaviour. It is the identifiability track.
- **A (NIST H I)** — the cleanest law-discovery problem with real data, and
  precise enough that the law's own *failure* (QED) is visible.
- **R (DE441)** — two extremes in one source: Kepler, where the extrapolation
  is 20× in `a`, and the 1PN correction, which is 8×10⁻⁸ of the signal and
  therefore a numerics problem before it is a physics problem.

## Why the arms are what they are

**`oracle` is not a competitor.** It has zero free physics parameters and
exists to bound the others. When an arm beats it, that is a claim about the
*published constant*, not about the method — and on track A the claim is
correct: the oracle carries Bohr's `R_H`, which the data says is 10.8 ppm low.

**`physics` before `pinn`.** The classical parametric inverse problem is the
right baseline for "physics-informed", and most of the time it wins. A PINN
that cannot beat `curve_fit` on a one-parameter law is not evidence of
anything; reporting `physics` separately keeps that visible.

**The PINN is `law + σ_y·NN` with a weight on the correction**, rather than a
plain residual PINN, for three of the four tracks. This makes "how much
physics" a *continuous dial* that spans both endpoints exactly: `w_phys → ∞`
reproduces `physics`, `w_phys = 0` is a black box with a physics-shaped
initialisation. Track G uses the residual PINN of Raissi et al. instead,
because its law is a differential equation and that is the natural form.

**The black box is tuned.** Width, depth and weight decay are grid-searched on
held-out data on the tuning seeds and frozen. The single most common way to
overstate the value of a physics prior is to compare against an untuned MLP.

## Why the noise sweep scores against clean targets

The arms are trained on corrupted values and evaluated against the *clean*
measurements. Scoring against the corrupted values instead drives every arm
toward the same noise floor and hides the effect the sweep exists to measure,
which is *who recovers the signal*. This was changed after the first run: the
initial version scored against the noisy targets and the curves were nearly
flat.

## Why law recovery is measured, not parsed

PySR rarely returns `c·f**p`. It returns `f*0.625*f**1.478/f**(-1.188)`, which
is the same function. So the exponent is obtained as the slope of log y
against log x, with a constancy check that returns `None` when the expression
is not a power law at all. The same principle applies on track Q: the question
"is this Planck's law?" is answered by evaluating the expression against the
exact Planck function inside and outside the fitted band, not by inspecting
its text.

An earlier version of that check compared the low-frequency log-slope against
2 and declared the answer wrong. The exact Planck function's slope over the
same interval is 1.84, because `x = hν/kT` is not small enough there for
Rayleigh-Jeans to hold. **The check was wrong, not the answer.** It is
recorded here because it is exactly the failure mode the project is about:
a plausible-looking criterion that measures the wrong thing.

## Why track R has a convergence study rather than a result

The GR term is 8×10⁻⁸ of Mercury's acceleration. A 4th-order derivative at a
3-hour step carries a truncation error of a few ×10⁻⁹ — a few per cent of the
signal — and it is *correlated with the orbit*, so it does not average out. It
lands almost entirely in `α`:

| stencil | step | α |
|---|---|---|
| 2nd order | 3 h | ~10³ |
| 4th order | 6 h | 3.15 |
| 4th order | 3 h | 1.1343 ± 0.0024 |
| 4th order | 1.5 h | 1.008 |
| 6th order | 3 h | 1.00012 ± 0.00002 |

Run once at the third row and stopped, this project would have reported a
13.4% violation of general relativity at 56 formal sigma. The error bar was never
the problem; it was correct, and it was answering a question about the
*statistics* of a model that was *systematically* wrong.


## The Phase 2 ablation, and a prediction that was wrong

`docs/plans/PLAN.md` section 3.2 ranked the candidate PINN options by expected
value before any of them were run. Early stopping was ranked first -- "largest
single win on kepler/hydrogen/cmb" -- on the reasoning that the correction
network overfits wherever the published law is already right, and that
stopping it early would be the direct fix.

The tuning seeds say the opposite. Early stopping **hurt** four of eight
cells and helped none; gradient-norm loss balancing, ranked fourth, was the
largest effect on the board, worth up to 101x on hydrogen extrapolation.
Fourier features, ranked fifth with the note that they "may hurt the three
smooth tracks", hurt five of six.

The reasoning behind the ranking was not obviously wrong, and it is recorded
here rather than quietly reordered, because the ranking is exactly the kind
of plausible argument this project exists to distrust. The ablation was run
one switch at a time precisely so that the answer would not depend on it.

A second thing the same run caught: the shipping rule in `docs/plans/PLAN.md`
section 5.2 said an option ships if it "helps on at least one track and on no
track improves in-distribution error while degrading parameter recovery".
Fourier features satisfy that -- they help kepler extrapolation -- while
making three tracks up to 98x worse. The rule had never considered an option
that *hurts*, and now requires that it hurt none.

## Seeds

Tune on 3 / 7 / 19, report on 11 / 23 / 42. Hyperparameters, the SR operator
sets and the finite-difference stencil were all chosen without looking at a
reporting seed.

## What is cached, and what that means for reproducibility

Symbolic searches are cached on disk under `.cache/sr/`, keyed by a hash of
the data *and* the search configuration. Deleting the cache and re-running
reproduces the results because PySR is run with `deterministic=True`,
`parallelism="serial"` and an explicit `random_state`. The sweeps use a
shorter search (25 iterations) than the headline fits; every row records
which it used.

---

# Part II — the simulations, and what they changed

The project originally had only real data. Adding simulations was not
decoration: three results below exist *only* because a controlled case with a
known answer was available.

## Why simulate at all when real data is the point

A real-data result has two unknowns stacked on top of each other: the physics
and the method. A simulation removes one of them. When the law is exactly what
was put in, whatever a method fails to recover is the **method's own error**,
and that number is the floor every real-data result has to be read against.

Concretely: symbolic regression recovers the gravitational exponent as
−1.9999969 and `GM` to 0.6 ppb from a simulated orbit. So when the same
machinery returns `GM_sun` from the real ephemeris with a tens-of-ppm
residual, that residual cannot be blamed on the method — it is the two-body
formula neglecting the planets' masses.

## The SNR threshold was wrong, and only an injection could show it

The GW150914 frequency extraction keeps cycles whose envelope SNR clears a
threshold. It was set to **2.0** a priori. It looked fine, the real track
looked fine, and the chirp mass it produced agreed with GWTC-1 at 2PN.

Injecting a waveform with a **known** chirp mass into the **real detector
noise** and running the identical code:

| SNR cut | median recovered `Mc` (true 31.17) | bias | trials within 20% |
|---|---|---|---|
| 2.0 | 9.3 | −70 % | 0 / 10 |
| 2.5 | 29.4 | −5.7 % | 6 / 10 |
| 3.0 | 30.9 | −0.9 % | 9 / 10 |
| 3.5 | 33.2 | +6.4 % | 5 / 10 usable |

At 2.0 the extraction admits noise-induced extra zero crossings that read as
200 Hz where the truth is 40 Hz; a handful of those wreck the fit. The
threshold was re-chosen **on injections only**, and the real event
re-measured with it. The real conclusions survived — 0PN still biased by about
+9 M☉, 2PN still consistent with GWTC-1 — but they survived *as a result*
rather than as luck.

This is the seed discipline (`tune on 3/7/19, report on 11/23/42`) extended to
a signal-processing choice. A pipeline nobody has injected into has no error
budget, and a constant it returns is a number with no uncertainty attached to
the method that produced it.

## Two bugs the injection found in the simulation itself

- **The waveform had no ringdown.** It stopped dead at 250 Hz, so the smoothed
  envelope of the band-passed result peaked in the *middle* of the inspiral.
  Since the extraction keeps only cycles before the envelope peak, the
  injection was silently discarding its own highest-frequency cycles and
  returning 13.7 M☉ for an injected 31.2. Fixed by appending a damped tail.
- **Flat noise is not the detector's noise.** Whitening divides by the
  amplitude spectral density, which boosts the 100–200 Hz band where the
  informative high-frequency cycles live. Injecting into white noise changes
  which cycles survive. The injection now goes into the real 32 s record, at a
  quiet offset, with L1 inverted and advanced 6.9 ms so the project's own
  conditioning recombines the two copies exactly as it does for the real
  signal.

## A small data loss does not buy a right constant

The PDE rung of the neglected-terms study recovers a diffusivity from
`u_t = alpha u_xx + C(u, u_x)`. At `eps = 0` the modelled law is exactly
right and `alpha` must come back 0.05. It came back **0.015** — 70 % wrong —
and stayed wrong through every fix that operates on the loss:
gradient-norm balancing, a data-only warmup, a hard initial condition through
a saturating gate, a curriculum that opens the time horizon gradually,
residual-adaptive collocation, and capping the residual weight. Each improved
the fit. None moved `alpha` out of 0.015–0.033.

They could not, because the loss was not where the error was. The measurement
that settled it (`benchmark.neglected.derivative_accuracy_study`) fits the
same network to the data **alone**, with no residual term at all, and then
reads the derivatives off the result:

| | mean \|u_t\| | mean \|u_xx\| | implied `alpha` |
|---|---|---|---|
| exact solution | 0.190 | **3.79** | **0.05000** |
| network, data loss 8.4×10⁻⁴ | 0.201 | **6.09** | 0.0334 |

The field is excellent and `u_t` is right to 6 %. The **second derivative is
60 % too large**. Since `alpha` is exactly the ratio `|u_t| / |u_xx|`, it
lands a third low before any physics term has spoken.

Three controls say the excess is the network's own and not the data's:

- at **zero noise** the error is unchanged (35 %), so it is not noise being
  differentiated;
- at **eight times the data** it is slightly *worse* (39 %), so it is not
  sparsity;
- the least-squares estimator itself returns 0.05000 on the analytic field,
  so it is not the estimator (`test_implied_alpha_is_exact_on_the_exact_field`).

What remains is the general fact: **a network's accuracy in the k-th
derivative is not controlled by its accuracy in the value.** High-frequency
content costs almost nothing in function space and is amplified by every
differentiation. This is why a PINN can report a small residual and a wrong
constant at the same time — and why "the residual converged" is not evidence
that the physics was identified.

The remedy that follows from the diagnosis is to penalise the wiggle one
derivative **above** the one the equation reads, so the term the residual
depends on is the one being controlled:

```python
u_xxx = torch.autograd.grad(u_xx.sum(), x, create_graph=True)[0]
smooth = torch.mean(u_xxx ** 2) / (scale / ic_width ** 3) ** 2   # dimensionless
```

Nondimensionalising by the initial profile's own scale is what makes the
weight portable; without it the penalty means something different at every
amplitude.

**It works in isolation and largely fails in place, and both halves are the
result.** Applied to the data-only fit, where `alpha` is read off the field by
least squares, it takes mean `|u_xx|` from 5.46 to 3.60 against a true 3.79
and the error from 35 % to 11 %. Applied inside the full fitter, where `alpha`
is instead *trained by gradient descent through the residual*, the weight
sweep on the tuning seeds 3/7/19 (`results/neglected/tune_pde_smooth.csv`)
moves it barely at all:

| `w_smooth` | recovered `alpha` (true 0.05) | error |
|---|---|---|
| 0 | 0.01202 ± 0.00129 | 76.0 % |
| 0.003 | 0.01278 ± 0.00056 | 74.4 % |
| 0.01 | 0.01368 ± 0.00050 | 72.6 % |
| 0.03 | 0.01535 ± 0.00072 | **69.3 %** |
| 0.1 | 0.01610 ± 0.00217 | 67.8 % |

Monotone, consistent across all three seeds, and far too small: a 33× range of
the weight buys 8 percentage points. `w_smooth = 0.03` ships — it helps, hurts
nothing, and the mechanism behind it was measured rather than guessed — but
0.1 is not chosen despite its better mean, because its seed spread is four
times larger and a difference that size is not resolved by three seeds.

So excess curvature is *a* real cause and not *the* remaining one.

### The correction network was the obvious suspect, and it is innocent

The hypothesis was that the free correction is what destroys
identifiability: with `C(u, u_x)` in the residual, the pair `(alpha, C)` is
not identified at all, because for **any** `alpha` there is a `C` satisfying
`u_t - alpha u_xx - C = 0` exactly, leaving only the small penalty
`w_phys·mean(C²)` to pin `alpha` down. The least-squares reading has no such
freedom, which would explain why it responds to the curvature fix and the
trained arm does not.

It was tested at `eps = 0`, where the true correction is **exactly zero**, so
removing `C` costs nothing and the test is clean
(`results/neglected/tune_pde_correction.csv`):

| | recovered `alpha` (true 0.05) | error |
|---|---|---|
| `C` free, `w_phys = 1e-3` | 0.01535 ± 0.00072 | 69.3 % |
| `C` penalised, `w_phys = 1e0` | 0.01534 ± 0.00100 | 69.3 % |
| `C` penalised, `w_phys = 1e3` | 0.01478 ± 0.00028 | 70.4 % |
| **`C` removed entirely** | **0.01406 ± 0.00097** | **71.9 %** |

**Refuted.** A 10⁶ range on the penalty and then deleting the term outright
moves `alpha` by 0.0013, in the *wrong* direction. The reasoning was sound
and the conclusion was wrong, which is the only useful kind of negative
result: it removes a suspect instead of adding a caveat.

### The optimiser was never the problem

`alpha` is exactly `argmin |u_t - alpha u_xx|²`, so **any** field implies an
`alpha` by least squares whether or not one was trained. Training the full
residual PINN and then asking that same field what `alpha` it implies
separates two questions that had been run together
(`physprior neglected consistency`,
`results/neglected/tune_pde_consistency.csv`):

| | trained `alpha` | its own field implies | ratio |
|---|---|---|---|
| `C` removed, seed 3 | 0.01470 | 0.01463 | **1.00×** |
| `C` removed, seed 7 | 0.01480 | 0.01474 | **1.00×** |
| `C` removed, seed 19 | 0.01269 | 0.01260 | **0.99×** |
| `C` free, seed 3 | 0.01621 | 0.00759 | 0.47× |
| `C` free, seed 7 | 0.01541 | 0.00786 | 0.51× |
| `C` free, seed 19 | 0.01445 | 0.00586 | 0.41× |

Derivatives are central differences at two step sizes; the drift is ~4×10⁻⁶,
four orders below the effect (invariant 5).

**With `C` removed the optimiser is exactly self-consistent.** It returns the
least-squares `alpha` of the field it produced, to three digits, on every
seed. It is doing precisely what it was asked to do. Whatever is wrong is
wrong with the **field**, and nothing addressed to the loss weights, the
schedule, the sampler or the parameterisation can reach it — which is why
none of them did.

(With `C` free the ratio is ~0.45, because the correction absorbs the
discrepancy and breaks that consistency. It lands nearer the truth than the
field supports, by accident rather than by identification — a reminder that a
closer number is not a better method.)

### So what is wrong with the field

| the field | implied `alpha` (true 0.05) |
|---|---|
| fitted to the data alone, no residual | 0.032 |
| after full residual training | 0.0147 |

**Switching the physics term on makes the field roughly twice as bad at
identifying the constant as having no physics term at all.** The residual is
self-defeating here: it degrades the very field it needs in order to identify
`alpha`, and `alpha` then faithfully tracks the degraded field.

That also explains the split that had looked paradoxical. The curvature
penalty takes the data-only reading from 36 % wrong to 6.8 % and the trained
arm only from 76 % to 68 %, because it fixes curvature and does nothing about
the residual wrecking the fit — the data loss goes from 8.4×10⁻⁴ to order 1
once the physics term is enabled.

The accurate statement of the failure is therefore **not** "the PINN did not
converge". It is: *on this problem, in this configuration, the physics term
costs more field accuracy than the physics constraint buys.* Whether that is
fixable — by a different residual scaling, by solving for `alpha` in closed
form from a data-only field, or by a formulation that does not put `u` and
`alpha` in the same optimisation — is open, and the arm stays
`converged=False` until something demonstrates it.

**The arm therefore stays marked `converged=False`.** A diagnosis is not a
result, and 69 % is not a recovered constant.

## Perihelion location: root-finding, not parabola-fitting

Mercury's GR shift is 5×10⁻⁷ rad per orbit. Locating the perihelion by fitting
a parabola to a sampled `u(φ)` grid is good to about 10⁻⁶ rad — *bigger than
the effect*, and the first version of this simulation reported a Newtonian
"precession" 20% larger than the GR one. Finding the root of `du/dφ` on the
solver's dense output instead reaches ~10⁻⁹, and the simulated precession then
matches `6πGM/(c²a(1−e²))` to six significant figures, stably across
tolerances and orbit counts.

Same lesson as the ephemeris track's finite-difference stencil, reached from
the other direction.

## Richardson extrapolation, because the solver was coarser than the effect

Plain second-order finite differences give the hydrogen levels to ~300 ppm.
The relativistic + QED shift in the real atom is 10.8 ppm. A solver 30× less
accurate than the effect cannot see it, and differencing anyway would have
reported discretisation error as physics. Two grids combined as
`E + (E_fine − E_coarse)/3` reach 4 ppm, and the comparison becomes meaningful
— it gives +15 ppm against NIST, agreeing with the fitting route's +10.8 ppm
within the solver's remaining error.

The `r_min` sweep in the same module is the companion warning: at
`r_min = 1e-3` the error stops responding to the grid entirely. Refining `dx`
does nothing, because the limit is the inner boundary. A convergence study in
only one parameter would have missed it.

## Symplectic integration, and why RK4 is kept

`integrators.py` ships both velocity Verlet and RK4 not because a choice was
left open but because the comparison is the teaching point. RK4 has the better
local error and the worse long-time behaviour: at 4-day steps over 200 years
Verlet's energy error stays at 2% while RK4's reaches 412% and the orbit
unbinds. The integrator is part of the physics model.

## What the simulations do NOT do

They do not validate the physics — the laws are put in by hand. They bound the
method. Any claim in this project of the form "the data says X" rests on the
real data; the simulations only say how much of "X" could have been the
method talking.
