# PHASE 6 PLAN — quantum interference tracks

Written at the gate `phase6.md` stop point 1, which requires this document
**and** the Q1 real-data verification before approval. Both are below.

## 0 · Gate status

| precondition | state |
|---|---|
| Phase 2 merged, PINN default frozen on 3/7/19 | **yes** (updated 2026-09-26) — merged into `main`; `FROZEN_PINN = PinnOptions(balance=True)` in `src/physprior/methods/pinn.py` (replaced 2026-09-27 by a tuned per-track weight; `docs/DECISIONS.md`). `balance+ens5` was measured and not shipped ([`PLAN.md`](PLAN.md) §5A) |
| Phase 5 approved | **no** — `docs/plans/PHASE5_PLAN.md` is at its own gate |
| Q1 real-data verified | **yes — done below, and the answer is negative** |

Phase 6 depends on Phase 5 only through Q4, which feeds the Phase 5 Part A
loss search. Q1, Q2 and Q3 do not. So Phase 6 can run before Phase 5 if the
interference tracks are wanted first — that is a real option,
because Q1 needs no new data loader at all.

---

## 1 · Q1 real-data verification (required before approval)

The spec says: do not assume a dataset exists; digitising a printed figure
fails the provenance invariant. All three candidates were checked.

| candidate | what exists | verdict |
|---|---|---|
| **Bach et al. 2013**, *Controlled double-slit electron diffraction*, NJP 15 033018 | Open access, and there IS supplementary material — but it is a **movie** of the electron build-up (`stacks.iop.org/NJP/15/033018/mmedia`), not a table. IOPscience additionally serves automated requests through a bot-validation redirect, so even the listing is not machine-fetchable | **fails** — extracting intensities from compressed video frames is digitisation with worse provenance than a figure: unknown normalisation, unknown compression |
| **Tonomura et al. 1989**, *Demonstration of single-electron buildup*, Am. J. Phys. 57 117 | The result is a 1989 film from Hitachi. No public archive of the frame data is discoverable; the searchable record is the paper and the movie | **fails** — no archive, no checksum, no URL |
| **Jönsson 1961 / Zeilinger 1988** | As the spec anticipated, published as figures | **fails** |

**Conclusion: Q1 ships simulation-only**, which `phase6.md` explicitly allows.
The track's value is the Fresnel dial, and that needs
a controlled truth regardless. `docs/RESULTS.md` will say so in those words.

This also means **Q4 is the only interference track with a real-data path**,
and it has none of its own either: its "real" anchor is the Poisson counting
statistics, which are a property of the simulation. Worth being explicit, since
`phase6.md` §0 describes Q4 as "the real-data anchor Phase 5 Part A needs" —
it is an anchor in *noise model*, not in measured data.

---

## 2 · Q1's analytic claims, checked before any track code was written

`phase6.md` makes two hard predictions and asks for them as tests. Both hold,
verified numerically at `a = 30 µm`, `d = 90 µm`, `λ = 500 nm`, `L = 1 m`:

**Fisher information rank.** Scaling the columns by the parameters and taking
the SVD of `JᵀJ` in `(a, d, λ)`:

```
s0 = 6.597e+02        s1/s0 = 7.9e-02        s2/s0 = 1.04e-16
rank(F) = 2
null vector = (-0.57735, -0.57735, -0.57735)
|cos| with the scaling direction (1,1,1)/sqrt(3) = 1.0000000000
max |I(theta) - I(1.37 * theta)| = 5.0e-16
```

Rank 2, not 3, and the null direction is exactly the scaling direction — the
pattern depends on `(a, d, λ)` only through `a/(λL)` and `d/(λL)`. The
prediction is confirmed to machine precision.

**Missing orders at `d/a = 3`.** Interference orders `m = 3, 6, 9` are
suppressed to `1e-32` while their neighbours sit at `1e-2`:

```
m :        1         2         3         4         5         6         7         8         9
I : 6.84e-01  1.71e-01  1.05e-32  4.27e-02  2.74e-02  1.05e-32  1.40e-02  1.07e-02  1.52e-33
```

Both go in as analytic tests computed in code, per the spec and invariant 1.
Doing this now rather than at stop point 2 means the track's central claim is
already de-risked: if it had failed, the phase would not be worth starting.

---

## 3 · Per-track assessment

### Q1 `double_slit_far_field` — cheapest, highest value, start here

No loader, no network, no new dependency. The work is the Fresnel/
Rayleigh–Sommerfeld integrator (with the quadrature convergence study invariant
5 requires) and the `Problem`. The Fresnel-number crossover is a genuinely new
kind of result for this repo: a *physical* dial for prior wrongness, where
`w_phys` is a hyperparameter dial. It is also a direct test of the Phase 0
diagnosis — that the PINN pays exactly where the law is incomplete — with the
incompleteness now tunable rather than accidental.

### Q4 `interference_counts` — do with Q1, as the spec says

Shares Q1's simulation. The substantive addition is the **asymmetric** loss
family: Poisson deviance `mu - n log mu` cannot be represented by Phase 5's
symmetric spline, so Part A needs a second family. Note this makes Q4 a
*dependency of* Phase 5 Part A's design, not just a consumer of it — if Phase 5
lands first with a symmetric-only family, it will need extending anyway. That
argues for doing Q1+Q4 **before** Phase 5 Part A, contrary to the stated order.

### Q3 `visibility_duality` — the conceptually new one, and cheap

The inequality prior (`relu(D² + V² − 1)²`) is a genuinely different kind of
physics term from anything in the repo, and the control is excellent: inject
data violating the bound and check the arm *refuses* to fit it. That control is
the whole value — it is the only proposed experiment that can catch a physics
term which is silently non-binding. Low compute, 1D, no new data.

### Q2 `tdse_2d_slit` — expensive; see §4

---

## 4 · Q2 compute estimate (stop point 3, answered early)

Measured baselines from this repo: a 1-D PINN fit at 6000 epochs on ~40 points
takes **~25 s** on this machine (from the `seconds` column of the committed
sweeps); the whole offline test suite is 11 s.

| component | estimate | basis |
|---|---|---|
| 2D split-operator ground truth, 512² grid, ~2000 steps | ~10 s per run; convergence study in `dx`, `dt`, domain, barrier smoothing width ≈ 20–30 runs → **~5 min** | FFT cost, measured 1-D solver scaled |
| 2D residual PINN, ~10⁴ collocation points, complex `psi` as two outputs | **~40–100 min per fit** — 10²–10³× the 1-D per-epoch work | scaled from the 25 s baseline |
| ablation grid: {baseline, fourier, causal, RAR} × 3 seeds | 12 fits → **8–20 hours** | above |

Q1 + Q4 + Q3 together are minutes-to-a-few-hours. **Q2 alone exceeds them by
roughly an order of magnitude**, which by stop point 3 needs a separate decision.
Recommendation: land Q1, Q4 and Q3 first, then decide on Q2 with the
Fresnel crossover already in hand — it may well answer the spectral-bias
question more cheaply than the 2D solver would.

---

## 5 · Conflicts and risks

**No invariant conflicts.** As with Phase 5, the spec is stricter than
`CONTRIBUTING.md` in places (analytic tests computed in code, the FIM rank
assertion, the missing-orders structural check); its version is kept.

Three risks worth stating:

1. **The ordering in `phase6.md` §0 vs Q4's dependency on Phase 5.** Q4 needs
   an asymmetric loss family that Phase 5 Part A as specified does not have.
   Either Q1+Q4 come first and Part A is designed with both families from the
   start, or Part A ships symmetric-only and is revisited. The former is
   recommended.
2. **Q2's absorbing boundary sets the floor on every conservation
   diagnostic.** The spec already says to report how much norm leaves the
   domain. That number must be measured *before* the Ehrenfest results are
   interpreted, or a boundary artefact will read as a physics violation —
   precisely the Mercury stencil failure in a new costume.
3. **`Problem` and the two-identifiable-combination scoring.** Parameter
   recovery is currently scored per named parameter against
   `theta_published`. Scoring on *combinations* (`a/(λL)`, `d/(λL)`) needs a
   small extension: a track must be able to declare derived quantities to
   score. This is a real change to `benchmark/protocol.py`, not a track-local
   detail, and it is the one piece of Phase 6 that touches shared code.

---

## 6 · Proposed order, across both phases

Given §3 and §4, and that Phase 5's two real-data tracks are its long pole:

| # | work | why here |
|---|---|---|
| 1 | freeze Phase 2 defaults | everything is read against them |
| 2 | **Q1 + Q4** | no data dependency, gives the Fresnel crossover and the Poisson-vs-MSE result, and fixes Part A's loss-family requirements |
| 3 | **Q3** | cheap, and the only test of a non-binding physics term |
| 4 | Phase 5 Part A, both loss families | now specified correctly by Q4 |
| 5 | Phase 5 B1 `em/rain_attenuation` | the long pole: 318 MB archive, wet-antenna and wet/dry calibration |
| 6 | Phase 5 B2, then B3 | — |
| 7 | Q2, if the compute is approved | — |

## 7 · Decisions required

1. **Approve Phase 6** (stop point 1), with Q1 simulation-only as verified.
2. **Order**: §6 as proposed, or `phase5.md`'s original A → B1 → B2 → B3 then
   Phase 6.
3. **Q2**: run it, or defer until the cheap tracks have reported.
4. Still outstanding from `docs/plans/PHASE5_PLAN.md` §5: the Mie question, and what
   "organise by topics" should change.
