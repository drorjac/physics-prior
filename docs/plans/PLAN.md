# PLAN — repositioning physprior around the PINN arm

Phase 0 audit and the plan it implies. Every number below is from `results/`
on the **reporting seeds (11/23/42)**, computed by
`scripts/audit_pinn.py`. Nothing here is an estimate.

---

## 1 · Audit findings

### 1.1 Working tree

`notebooks/00_overview.ipynb` and `notebooks/01_gravity.ipynb` are modified.
The diff is **metadata only** — `ExecuteTime` keys and JSON key reordering
from a JupyterLab run on 2026-09-22. Cell *sources* are byte-identical to
`HEAD` (9 and 24 cells, outputs present in both). Nothing of substance is at
risk, but Phase 1 rewrites these files, so they are committed on a branch
before it starts.

### 1.2 Baseline is green

| check | result |
|---|---|
| `make test-offline` | **82 passed**, 8 deselected, 12.4 s |
| `ruff check` | clean |
| `ruff format --check` | 68 files already formatted |
| `mypy` | no issues in 55 source files |

(Run with `PYTEST_ADDOPTS=-p no:faulthandler`; see §5.3.)

### 1.3 The α "inconsistency" is a rounding defect, not a stale number

The brief flagged `α = 1.13 ± 0.002` against "56 formal sigma", since
`(1.13−1)/0.002 = 65`. **Neither side of the claim is stale.** From
`results/relativity/mercury/gr_convergence.csv`, the 4th-order / 180 min row:

```
alpha_GR = 1.1343426790590472    alpha_sigma = 0.002403694027386392
(α − 1)/σ = 55.89  →  56 formal sigma        deviation = +13.43 %
```

So **56σ is correct**. What was wrong is the *rounding in the prose*: α was
printed to 2 dp and σ to one significant figure, and the printed pair no
longer reproduces the printed conclusion. A reader doing the division gets 65.

Fixed by quoting `α = 1.1343 ± 0.0024` and `13.4 %` in `README.md`,
`docs/METHOD.md` and the project brief, and by two assertions in
`tests/test_claims.py`:

- `test_false_gr_violation` now pins α to 4 dp, σ to 4 dp and the deviation to
  1 dp against the CSV;
- `test_quoted_gr_numbers_reproduce_their_own_sigma_count` re-derives the
  σ-count *from the digits the README prints* and fails if they disagree by
  more than 1σ. It fails on the old wording (65 vs 56) and passes on the new.

This is the project's own invariant 1 applied one level deeper: it is not
enough for a quoted number to be traceable to `results/`; the quoted numbers
must also be self-consistent *as printed*.

### 1.4 A second, unreported instance of the same failure mode

The converged row is `α = 1.00012 ± 0.0000165` — which is **7.3σ from 1**, and
the 90 min / 6th-order row is `1.0000985 ± 0.0000128`, **7.7σ from 1**. Between
those two steps α moves by `2.2e-5`, i.e. **more than its own error bar**.

By invariant 5 ("a result still moving with step size is not a result"), the
6th-order α is *not yet converged either* — its formal error underestimates
the remaining truncation error by roughly a factor of 2. The README's claim is
"agrees with Einstein to one part in 10⁴", which is true and safe. But the
error bar attached to it is not the honest uncertainty. **Proposed:** report
the converged α with a systematic term from the step-halving difference
(`α = 1.00012 ± 0.0000165 (stat) ± 0.000022 (step)`), and add the convergence
assertion to `test_claims.py`. Flagged rather than fixed: it changes a
published number and needs your call.

---

## 2 · Where the PINN actually stands

Reporting seeds, mean ± std over 11/23/42, against the best **non-oracle**
arm, on **held-out** error (`nrmse_out`; see the note below). "TIE" means the
gap is inside the pooled seed-to-seed spread.

| track | interpolation | extrapolation | noise (max) | parameter recovery |
|---|---|---|---|---|
| `gravity/kepler` | TIE vs `physics` (4.0×) | **LOSES** vs `physics` 3.9× | **LOSES** vs `physics` 5.5× | ties `physics` (0.00586 % vs 0.00575 %) |
| `relativity/gw150914` | TIE vs `physics` (1.1×) | **WINS** vs `nn` (0.75×) | TIE vs `physics` (1.3×) | ties `physics` (−5.64 % vs −5.02 %) |
| `quantum/hydrogen` | **LOSES** vs `sr` 80× | **LOSES** vs `sr` 3.4e4× | **LOSES** vs `physics` 3.3× | ties `physics` (0.00117 % vs 0.00109 %) |
| `quantum/cmb` | TIE vs `physics` (1.3×) | **LOSES** vs `physics` 12.5× | **LOSES** vs `physics` 3.4× | ties `physics` (−0.0172 % vs −0.0171 %) |

**Score: 1 win, 7 losses, 4 ties out of 12 cells, and not one parameter
recovered better than `curve_fit`.** The single win is `gw150914`
extrapolation — the one track whose law is a truncated expansion.

> **Correction, recorded because it is the project's own failure mode.** The
> first version of this table scored interpolation and noise on `nrmse_in`.
> In `sweep_budget` and `sweep_noise`, `score()` is called with the *training*
> indices as `idx_in`, so `nrmse_in` is the fit to the data the arm was
> handed, not a held-out result. On that metric `physics` "beat" `sr` by
> **4.7e6×** on `gw150914` data efficiency — which is one parameter fitted
> through two points, i.e. exact interpolation, not a win. It also promoted
> the PINN's `cmb` interpolation tie into a win. `report.py:221` already used
> `nrmse_out` for the budget sweep; the conclusions module and
> `scripts/audit_pinn.py` now do too. A plausible-looking criterion that
> measures the wrong thing, caught by the number being absurd.

### 2.1 Diagnosis — this is not a tuning problem

Three patterns, all consistent:

1. **The correction network is pure variance wherever the law is already
   right.** On `kepler`, `hydrogen` and `cmb` the published law matches the
   data to within measurement precision, so the ideal `NN(x)` is zero. The
   network cannot represent zero exactly, so it contributes error that is
   invisible in-range (where data pins it) and explodes out-of-range (where
   nothing does): out/in ratios of 431× on hydrogen and 105× on cmb, against
   0.47× and 4.3× for `physics`.
2. **It pays exactly where the law is incomplete.** `gw150914` is the one
   track whose law is a *truncated expansion*, and it is the one track where
   the PINN is the best non-oracle arm out-of-range (0.496 vs `nn` 0.66,
   `sr` 6.89, `physics` 13.4) — and its **only** decisive win anywhere in the
   table. The correction has something real to absorb.
3. **Parameter recovery is a dead heat by construction.** The PINN tracks
   `physics` to three significant figures on every track. At `w_phys = 1` the
   physics term dominates, the correction is small, and θ converges to the
   same least-squares optimum `curve_fit` finds — but *without* a covariance
   matrix, so it returns the same answer with **less** information.

**Implication for the repositioning.** On this evidence, "a PINN architecture
for physics problems" is a claim the current tracks do not support. What they
support is sharper and more defensible: *a PINN buys you out-of-distribution
accuracy exactly when the physics prior is incomplete, and costs you one to
four orders of magnitude when it is not.* That is a finding, and it dictates
the track roadmap in §4 — the honest way to make the PINN the headline is to
**add tracks where its mechanism applies**, not to tune it into winning races
it is structurally set up to lose.

---

## 3 · Phase 2 — PINN optimisation

### 3.1 Two items are already implemented

| brief item | status |
|---|---|
| 1 · nondimensionalise `x`, `y` | **done** — `Standardiser.fit(x, y)` in `fit_pinn`; the correction enters as `σ_y·NN(x_std)` |
| 1 · log-reparameterise positive θ | **done** — `ParamSet`, `value = init·exp(raw)` (`pinn.py:49`) |
| — | `CosineAnnealingLR` already present; Adam only |

Genuinely new: L-BFGS refinement, loss balancing, Fourier features, RAR /
causal weighting, deep ensembles, early stopping.

### 3.2 Expected value, given §2.1

Ranked by whether the mechanism addresses a loss we actually measured:

| rank | change | what it should fix | expected |
|---|---|---|---|
| 1 | **Early stopping on a train-carved validation split** | the correction overfitting where the law is already right — the direct cause of the out-of-range blowup | largest single win on kepler/hydrogen/cmb |
| 2 | **Deep ensembles (5)** | `±5e-05` on a `5.12e-05` mean on kepler: seed variance currently exceeds the effect being measured | shrinks spread; gives the PINN an uncertainty to set against `curve_fit`'s σ |
| 3 | **L-BFGS refinement after Adam** | hydrogen's non-monotone `w_phys` response, which looks like optimisation noise, not physics | moderate |
| 4 | **Loss balancing (Wang et al. 2021 / GradNorm)** | tracks where the two loss terms differ by decades | moderate — but see §3.3 |
| 5 | **Fourier features** | spectral bias — only `gw150914` has structure at multiple scales | small; may *hurt* the three smooth tracks |
| 6 | **RAR / causal weighting** | only applies to `fit_pinn_ode` (gw150914) and the new PDE tracks | defer to Phase 3 |

I propose implementing 1–3 first and reporting the ablation before touching
4–6, because 1 and 2 target the measured failure directly and 5 plausibly
makes three of four tracks worse.

### 3.3 ⚠ Invariant conflict — the `w_phys` default cannot be tuned on the existing sweep

The brief says "keep the `w_phys` sweep as the reference". The sweep is
`protocol.py:341`, and it runs on **`REPORT_SEEDS` (11/23/42)** — as do
`sweep_budget`, `sweep_noise` and `study_extrapolation`. That is correct for
*reporting* (README finding 4 is a reported measurement), but it means
**`sweep_physics_weight.csv` may not be used to choose a default**. Invariant
6 forbids selecting anything on a reporting seed.

This matters, because the existing sweep says there is real money on the table:

| track | `nrmse_out` @ `w_phys=1` (current default) | best in sweep | |
|---|---|---|---|
| `gravity/kepler` | 2.00e-4 | **2.05e-5 @ w=10** | ~10× |
| `quantum/cmb` | 2.22e-4 | 2.11e-4 @ w=1000 | flat |
| `quantum/hydrogen` | 6.31e-5 | 2.83e-5 @ w=1000 | non-monotone, ~2× |
| `relativity/gw150914` | 0.373 | 0.074 @ w=1000 — **but `err_Mc` degrades to −9.5 %** | tension |

So part of "the PINN loses on extrapolation" is a **default-tuning artefact**,
not an architectural limit — `w_phys = 1.0` is a hard-coded default
(`protocol.py:129`) that the repo has no record of ever having tuned.

**Resolution:** add a tune-seed sweep (`sweep_physics_weight(..., seeds=TUNE_SEEDS)`)
written to `results/<track>/tune/`, pick the default there, freeze it, and
leave the reporting sweep untouched as the published result. The last row is
worth keeping as a finding in its own right: on `gw150914`, the `w_phys` that
minimises out-of-range error is **not** the one that recovers the chirp mass —
best extrapolation at `w=1000` comes with a −9.5 % `Mc`, while `w=1` gives
+0.44 %. Accuracy and identifiability select different models, which is
finding 4 sharpened.

---

## 4 · Phase 3 — new tracks

The brief asks to start with `cosmology/pantheon` and `pde/burgers`. Given
§2.1 I recommend **changing that order**, and will not start until you decide.

| track | tests the PINN mechanism? | verdict |
|---|---|---|
| `quantum/helium` | **yes — directly.** Bohr's law is *wrong by design* for He I; the correction network has real structure to absorb. This is the `gw150914` mechanism with 100× more data and no signal-processing chain in front of it | **start here** |
| `gravity/pulsar_spindown` | **yes.** `n = 3` is the law; the measured braking index is `n < 3`; the gap is exactly what a correction term should capture, and ATNF gives hundreds of objects | **second** |
| `cosmology/pantheon` | partly — ΛCDM is *not* wrong at SN Ia precision, so this becomes a fourth "law already right" track. Valuable for the H0/Ω_m degeneracy (identifiability, finding 2), weak for the PINN | third |
| `pde/burgers`, `pde/allen_cahn` | different question — simulation-only, tests optimisation failure modes (shocks, stiffness), no real data, no competing arms in the current sense. Needs a `Problem` variant that scores a field, not a curve | fourth; scope it properly first |
| `astro/transit` | high value, highest cost — needs a new loader, limb-darkening, and instrument systematics | defer |

`helium` and `pulsar_spindown` are also the two that most directly support a
*PINN-headline* repositioning, because they are the cases where the answer
"use the law, fitted" is not available.

One warning on `pde/*`: the current `Problem` assumes `x: (N,d) → y: (N,)`
with a closed-form or ODE law and five comparable arms. A PDE track has no
`physics` arm and no `sr` arm in the present sense, so it either needs
`arm_impl` overrides that return `None` for the missing arms, or a documented
second protocol. That is a design decision, not an implementation detail —
which is why it should not be first.

---

## 5 · Working agreements

### 5.1 Branches

`phase0-audit` (this document + the α fix) → `phase1-notebook-conclusions` →
`phase2-pinn-ablations` → `phase3-tracks-*`. Notebook metadata committed
before Phase 1 touches them.

### 5.2 What "shipped" requires

A change goes into the default config only if, on the **tune** seeds, it helps
on at least one track, and on no track does it improve in-distribution error
while degrading parameter recovery. The final report is the
change × track × metric table the brief asks for, evaluated on reporting seeds
*after* the defaults are frozen.

### 5.3 Environment

`pytest` needs `-p no:faulthandler` here (sandbox `PermissionError` before
collection). The `Makefile` targets do not pass it; either export
`PYTEST_ADDOPTS` or add it to `addopts` in `pyproject.toml` — worth doing, and
it is a one-line change I have not made without asking, since it affects CI.

---

## 5A · Phase 2 outcome — what shipped, and what it bought

Decided on the tuning seeds (3/7/19) by `physprior tune`, then measured on the
reporting seeds (11/23/42) by re-running the full pipeline.

### The ablation

| option | tracks helped | tracks hurt | shipped | why |
|---|---|---|---|---|
| **`balance`** (Wang et al. 2021 gradient-norm annealing) | 5 | 0 | **yes** | the largest effect on the board, and the recovered constants move by under 2% |
| `ens5` (5-member deep ensemble) | 5 | 0 | no | real, but weaker than `balance` alone on every cell, and it multiplies all 210 pinn fits per reporting run by five — 1.5 h against 7.3 |
| `balance+ens5` | 5 | 0 | no | best numbers, 1.3–1.4× beyond `balance` alone, for that same 5× cost |
| `early` (early stopping) | 0 | 4 | no | **hurt**: cmb and hydrogen up to 2.9× worse |
| `fourier16` (Fourier features) | 1 | 5 | no | hydrogen interpolation **98× worse** |
| `lbfgs` | 0 | 0 | no | ran on every track; the proposal never lowered the loss, so the revert guard discarded it every time |

Two of the brief's seven items were already implemented and left alone:
inputs and targets are standardised by `Standardiser`, and positive constants
are optimised in log space by `ParamSet`.

### What the shipped change bought, on the reporting seeds

`results/phase2_before_after.csv`, generated:

| track | interpolation | extrapolation | noise (max) |
|---|---|---|---|
| `gravity/kepler` | 2.9× | 5.0× | 5.5× |
| `relativity/gw150914` | 1.0× | 1.0× | 1.0× |
| `quantum/hydrogen` | 9.7× | **100.9×** | 28.6× |
| `quantum/cmb` | 1.2× | 10.9× | 3.7× |

Median 4.35×, best 100.9×. The PINN's scorecard moves from **7 decisive
losses to 2** (both to `sr` on hydrogen); it now ties `physics` on kepler
noise and beats it on hydrogen noise and kepler extrapolation, though inside
the seed spread.

`relativity/gw150914` is unchanged by construction, and that is the honest
outcome rather than a gap: its arm is the ODE-residual PINN, a different
model, and it reports the option as **not engaged** instead of returning a
baseline number under the option's name.

### `w_phys` was not re-frozen

The tune-seed sweep confirmed the direction but milder than the reporting
sweep suggested — kepler 3.3×, hydrogen 2.3×, cmb and gw150914 ~1.1×. Since
`balance` makes the weight adaptive, a fixed per-track constant and the
balancing rule are alternatives rather than a stack, and `w_phys = 1.0`
survives as the balancing rule's starting value. The published `w_phys`
sweep, which is a reported measurement on 11/23/42, is untouched.

---

## 6 · Decisions needed before Phase 2 starts

1. **α error bar (§1.4)** — report the converged α with a step-size systematic,
   or leave the README claim as it stands?
2. **`w_phys` default (§3.3)** — approve the tune-seed sweep and re-freezing
   the default? This changes published extrapolation numbers on `kepler`.
3. **Track order (§4)** — `helium` + `pulsar_spindown` first, as argued, or
   `pantheon` + `burgers` as originally specified?
4. **Repositioning (§2.1)** — headline the PINN as asked, or headline the
   sharper claim the data supports ("a PINN buys OOD accuracy exactly when the
   prior is incomplete")? I can write the README either way, but the first
   needs the new tracks landed before it is true.
5. **`-p no:faulthandler` in `pyproject.toml`** (§5.3) — yes/no.
