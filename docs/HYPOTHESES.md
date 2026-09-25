# Hypotheses — the physical questions this project answers

Every experiment in the repository exists to answer one of the questions
below. Each is stated as a question about physics and learning. Each carries
what would **refute** it, and its status is updated whichever way the
evidence goes. Numbers live in [`RESULTS.md`](RESULTS.md) and the topic pages,
where [`tests/test_claims.py`](../tests/test_claims.py) re-derives them. This
page carries none, so it cannot drift.

| # | Question | Status |
|---|---|---|
| H1 | Inside the range the data covers, does knowing the law help? | **supported — barely** |
| H2 | Out of range, is the gain from knowing the law, or from the data pinning the law's constants? | **supported — the constants** |
| H3 | Does a learned correction help when the law is incomplete, and hurt when it is complete? | **refined in simulation; one real case; open** |
| H4 | Without the physics constraint, does a network absorb the physics and return wrong constants? | **supported** |
| H5 | Can a good fit hide a wrong law? | **supported, twice on real data** |
| H6 | Can symbolic regression find a law outside its operator vocabulary? | **supported (negative)** |

---

## H1 · Inside the data, knowing the law barely helps

**Question.** Given enough clean measurements across a range, does a model
that knows the physical law predict better *inside that range* than a
well-tuned network that does not?

**Evidence.** Across all four real-data tracks, the tuned `nn` arm is
competitive in-range. Claims to the contrary usually compare against an
untuned baseline. See finding 1 in the [README](../README.md).

**Refuted if** a track shows the physics arms beating the *tuned* network
in-range by more than the seed spread, with ample clean data.

## H2 · Out of range, what the law buys is identifiability

**Question.** When predicting outside the data, is the advantage from
*having* the law, or from the data being able to *pin down* the law's
constants?

**Evidence.** On the CMB blackbody, widening the fitted band toward the
Rayleigh–Jeans tail improves out-of-band prediction by orders of magnitude,
while in-band error gets worse. The law is the same throughout;
only how well the data constrain the temperature changes. GW150914 is the
counter-control. With a few faint early cycles, extrapolation defeats every
fitted arm, physics included. See the [quantum](quantum/) and
[relativity](relativity/) pages.

**Refuted if** a track with a correct law and a poorly constrained constant
still extrapolates well, or one with a well-constrained constant does not.

## H3 · A correction helps when what the law misses is distinguishable from it

**Question.** A physics-informed network writes the answer as *law + learned
correction*. When the law is incomplete, does the correction capture what is
missing and win out of range? When the law is already right, does it only
add noise?

**Refined in simulation.** The [neglected-terms study](neglected/) puts the
answer in sharper form. The correction helps when the missing piece is
**distinguishable** from the law. When the missing piece has the law's own
shape, the fit absorbs it into the law's constant. The curve then looks fine
and the constant is wrong, and the correction has nothing to find. This held
on an algebraic law, an ODE and (for the `physics` arm) a PDE.

**On real data** there is one supporting case. GW150914 is the only track
whose law is a truncated expansion (Newtonian vs post-Newtonian), and it is
the PINN's only decisive out-of-range win. On the three tracks where the law
is complete (Kepler, hydrogen, CMB), the correction costs accuracy out of
range, as H3 predicts. Mercury is a real-data example of the *degenerate*
case. The model left out two effects: the Sun's oblateness, and the Sun's
own motion in the n-body relativistic equations. Both are shaped enough
like the GR term that the GR coefficient α absorbed them. They have opposite
signs, so α looked *nearly* right. Putting both back takes α to 1 within its
error (see [relativity](relativity/)).

**Next test: helium** (`quantum/helium`, planned). Hydrogen's `E = −R/n²` is
exact for one electron. In helium the second electron screens the nucleus,
and each angular-momentum series is shifted by a *quantum defect*:
`E = −R/(n − δ_ℓ)²`. The defect is large for S states, whose electron
penetrates the core, and nearly zero for high ℓ. Its leading effect goes as
1/n³, a different shape from the law's 1/n², so it is **distinguishable**.
H3 therefore predicts:

1. the PINN beats the hydrogenic law when extrapolating to high n;
2. it does **not** beat the complete Rydberg–Ritz law;
3. its learned correction looks like a quantum defect: large for S, near
   zero for high ℓ. This is a check on *what* was learned, not only on the
   error.

**Then: pulsars** (`gravity/pulsar_spindown`, planned). A magnetic dipole
spinning in vacuum slows as `ν̇ ∝ −ν³` (braking index 3). Real pulsars
measure below 3, so the dipole law is incomplete.

**Refuted if** on helium the PINN fails to beat the hydrogenic law out of
range, or its correction shows no ℓ structure. Either result is reported at
the same size as a success.

## H4 · Without the constraint, the network eats the physics

**Question.** If nothing forces the correction to stay small, does the
network take over part of the physics, so that the law's constant comes out
wrong while the predictions still look right?

**Evidence.** At `w_phys = 0`, `GM_sun` from Kepler and `T_CMB` from FIRAS
come out wrong while held-out error barely changes (finding 4 in the
[README](../README.md)). The physics term is what makes the constant
*identifiable*. It is not an accuracy regulariser.

**Refuted if** a track recovers its constant at `w_phys = 0` as well as at
the default.

## H5 · A good fit does not diagnose a wrong law

**Question.** Can a model with the wrong law, or wrong numerics, fit the data
well and still return a confidently wrong physical constant?

**Evidence.**
- **GW150914.** The Newtonian law fits the chirp about as well as the
  post-Newtonian one, but it returns a biased chirp mass.
- **Mercury.** A 4th-order derivative produced a many-sigma "violation of
  general relativity" that was entirely numerical. After that was fixed, the
  fit was still excellent, while its coefficient silently absorbed two
  omitted physical effects (H3).

**Refuted if** goodness of fit were found to separate the right law from the
wrong one on these tracks.

## H6 · Symbolic regression cannot leave its vocabulary

**Question.** Can symbolic regression discover a law whose form needs an
operator it was not given?

**Evidence.** It rediscovers Rydberg's formula from NIST hydrogen levels,
but it does not find Planck's law in the FIRAS band. A controlled
band-widening experiment isolates the cause: the operator set is itself a
prior. See the [quantum](quantum/) page.

**Refuted if** SR recovers Planck's law without an operator set that can
express it.

---

Decisions about *which* experiments to run next are recorded in
[`DECISIONS.md`](DECISIONS.md).
