# Findings

The main results of the benchmark tracks, with the numbers behind them. The
tables are in [RESULTS.md](RESULTS.md), generated from `results/`; every number
quoted here is re-derived from `results/` by `tests/test_claims.py`. The
Lorenz study has its own page: [lorenz/](lorenz/README.md).

## The five findings

**1 · Inside the training range, with enough clean data, the black box is
competitive.** Physics buys little there. Any claim to the contrary is usually
a comparison against an untuned baseline.

**2 · Outside it the gap is orders of magnitude, but the mechanism is
identifiability, not the mere presence of a law.** On `quantum/cmb`,
out-of-band error falls ~56× as the fitted band reaches into the
Rayleigh-Jeans regime while the **in-band error gets worse**. On
`relativity/gw150914` (the counter-control), extrapolation from four faint
early cycles defeats *every* fitted arm, physics included. A physics prior
buys extrapolation when the data can identify its parameter, and not
otherwise.

**3 · Only the physics arms return something a physicist can argue with.**
And three times the argument was worth having:

- **hydrogen**: fitting Bohr's law to the NIST levels returns the measured
  ionisation limit to a few parts in 10⁹. That limit sits **10.8 ppm above**
  Bohr's prediction: relativistic and QED corrections to the 1s level. The
  same shift is reached independently by solving Schrödinger's equation and
  differencing against NIST. A black box fits the same levels and can say
  nothing about QED.
- **GW150914**: the Newtonian inspiral law recovers a chirp mass biased by
  **+9 M☉** with a formal error of 4.1: a confident wrong answer. Adding the
  1.5PN tail and 2PN terms removes the bias while the RMSE barely moves.
  *Goodness of fit does not diagnose a wrong law.*
- **Mercury**: the acceleration differentiated out of the ephemeris at a
  plausible step size gives a GR coefficient of **α = 1.1343 ± 0.0024**: a
  13.4% violation of general relativity at **56 formal sigma**. It is entirely
  finite-difference truncation error. With a 6th-order stencil α agrees with
  Einstein to one part in **10⁴**. *The result is not α; it is α once it has
  stopped moving.* What is left, **α − 1 = +1.2×10⁻⁴**, is not numerical:
  halving the step again moves α by only 2×10⁻⁵. It is **two omitted pieces
  of physics that nearly cancel**. Putting back the Sun's oblateness and spin
  moves α to +4.7×10⁻⁴. Replacing the one-body GR term with the n-body
  equations DE441 integrates moves it the other way, to −3.5×10⁻⁴. With both,
  α − 1 = **+1.8×10⁻⁶ ± 1.4×10⁻⁵**. That is a consistency check, not a new
  test of GR, because DE441 itself assumes GR: it shows the model is now
  complete ([relativity](relativity/)).

**4 · The network eats the physics if you let it.** `w_phys` is the weight on
the physics term in the PINN's loss, and it decides whether the recovered
constant means anything:

| problem | constant | error at `w_phys = 0` | error at `w_phys ≥ 0.01` |
|---|---|---|---|
| gravity | `GM_sun` | **19.5 %** | 0.005 % |
| quantum | `T_CMB` | **1.33 %** | 0.017 % |

At `w_phys = 0` the neural correction is free, absorbs the signal, and the
physical parameter drifts to whatever is left over, while the held-out error
barely changes. A PINN that fits well is not thereby measuring anything. The
physics term is not a regulariser you tune for accuracy; it is what makes the
parameter identifiable.

**5 · A pipeline you have not injected into has no error budget.** The
GW150914 chirp mass is extracted through a chain of signal-processing choices.
One, the envelope SNR a cycle must clear, was set to 2.0 a priori and looked
perfectly reasonable. Injecting a **known** chirp mass into the **real
detector noise** and running the identical code says otherwise:

| SNR cut | median `Mc` recovered (true 31.17) | bias | trials within 20% |
|---|---|---|---|
| 2.0 | 9.3 | **−70 %** | 0 / 10 |
| 2.5 | 29.4 | −5.7 % | 6 / 10 |
| **3.0** | **30.9** | **−0.9 %** | **9 / 10** |

The cut was re-chosen **on injections, never on the real event**. The same
injections then settle finding 3: at Newtonian order the injection is biased
**+25 %** and the real event **+29 %**; at 2PN, **−1.2 %** and **−1.3 %**. The
Newtonian bias is post-Newtonian truncation, not an artefact of the pipeline.


## The PINN arm's configuration

Each track's `pinn` arm now carries its own physics weight, chosen on the
**tuning** seeds (3/7/19) from a validation block cut from the top of that
track's training range, so the choice rewards a correction that carries past
the range it was fitted on without touching the reported test points. The
rule and every candidate's score are in `results/<track>/tune/w_phys_selection.csv`
([`benchmark/pinn_tuning.py`](../src/physprior/benchmark/pinn_tuning.py)).

Until 2026-09-27 the arm used gradient-norm loss balancing (Wang et al.
2021) instead. In this arm the physics term is a penalty on the correction,
whose gradient vanishes as the correction shrinks, so balancing raised the
weight without bound and switched the correction off: the arm was the
`physics` fit under another name ([`optimization/`](optimization/)
§5). The ablation that had shipped it measured that collapse as an
improvement over an unbalanced arm whose correction overfits.

With the correction switched back on, the results tables show what it does:
out of range it is worse than the fitted law on hydrogen, CMB and helium on
every reporting seed, and on Kepler in the median but not on every seed; and
where the validation block prefers no correction at all, the weight is
pinned at the top of its grid and the arm is the law by choice. The
validation block sits just above the fitted range while the test runs much
further, so the rule cannot see how far out a correction fails; that is a
limit of any choice made without the test data.

Rejected, with their measurements: early stopping hurt four cells and helped
none; Fourier features made hydrogen interpolation **98× worse**; L-BFGS ran
on every track and its proposal never lowered the loss, so the revert guard
discarded it every time. A five-member ensemble does help, and stacked with
balancing helps most, but only by a further 1.3–1.4× for five times the
compute on all 210 PINN fits of a reporting run.

`relativity/gw150914` does not move, and this is expected rather
than a gap: its arm is the ODE-residual PINN, a different model, and it
reports the option as **not engaged** instead of returning a baseline number
under the option's name.
