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
honest baseline for "physics-informed", and most of the time it wins. A PINN
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
