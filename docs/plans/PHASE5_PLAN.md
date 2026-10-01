# PHASE 5 PLAN: EM propagation, atmosphere, and loss discovery

Written at the gate `phase5.md` §0 defines: **Phase 2 is still open**, so this
document is the whole deliverable and no Phase 5 code is written until it is
approved (§5.1).

Status of the precondition (updated 2026-09-26; the table as first written
said "no" to both rows below):

| requirement | state |
|---|---|
| Phase 2 merged | **yes**: `phase2-pinn-ablations` is merged into `main` |
| PINN default frozen on 3/7/19 | **yes**: `FROZEN_PINN = PinnOptions(balance=True)` in `src/physprior/methods/pinn.py` (replaced 2026-09-27 by a tuned per-track weight; `docs/DECISIONS.md`); the ablation is in [`PLAN.md`](PLAN.md) §5A |
| Phase 5 numbers read against that frozen config | unblocked; Phase 5 itself still awaits approval (§5) |

---

## 1 · What fits the repo as it stands, and what does not

### 1.1 The three new tracks fit `Problem`, including B3

B3 was expected to hit the limitation recorded in `docs/plans/PLAN.md` §4, where a PDE
track has no `physics` arm and no `sr` arm in the present sense. It does not,
because the spec measures **probes**, not fields: with `x = (x_probe, t)` and
`y = E_z`, a 1D Maxwell track is an ordinary `Problem` with a 2-D input, the
`physics` arm is an FDTD forward model wrapped in `curve_fit` over `eps_r`,
and `sr` reads the probe time series. No second protocol is needed.

This is worth saying plainly because it changes the earlier recommendation:
`em/maxwell_1d` is **cheaper to land than `pde/burgers`**, which does need the
field-scoring redesign. If the PDE tracks are still wanted, B3 is the better
first wave problem.

### 1.2 Part A does not fit `Problem`, and should not be forced into it

Loss discovery is not a track. It has no `x → y` benchmark, the six protocol
questions do not apply to it, and its competitors are losses rather than arms.
It needs its own module (`physprior/benchmark/loss_discovery.py`), its own
results directory, and its own notebook. The one thing it should borrow
verbatim is the **oracle rule**: `-log p_true` is the ceiling, and a learned
loss that beats it out of sample needs an explanation recorded in `meta.json`,
exactly as `physprior report` already demands of the oracle arm.

### 1.3 What the existing machinery gives Phase 5 for free

- the Phase 1 conclusion machinery (`reporting/conclusions.py`): the new
  notebooks get verdict tables and generated prose by registering their tracks
  in `TRACKS_BY_PROBLEM`;
- the tune/report seed separation and `results/<track>/tune/`;
- `data/cache.py` provenance: URL, byte count, SHA-256 per file;
- `numerics/stencils.py` Richardson extrapolation, which B3's FDTD probe
  convergence study needs and should not reimplement.

---

## 2 · Conflicts and risks, in the order they can stop the phase

### 2.1 No conflict with the invariants: the spec anticipates them

Each instruction in `phase5.md` was checked against `CONTRIBUTING.md`, and
nothing needs overriding. Three places where the spec is *stricter* than
the invariants; its version is kept:

- §2.6 forbids optimising the loss on real data at all, not merely on a
  reporting seed;
- B1 calibrates the wet/dry threshold on injections only (the GW150914
  precedent, applied before the failure rather than after it);
- B2 fixes the station and date rule before any profile is read.

### 2.2 Data availability is the largest risk, and it is not resolvable from here

Both real-data tracks depend on archives whose canonical URL and checksum the
spec requires to be verified directly:

| track | archive | risk |
|---|---|---|
| B1 | OpenMRG (Gothenburg CML + gauges) | a Zenodo-style record; the exact record id and file layout must be pinned. CML datasets are commonly redistributed, so the wrong copy is easy to fetch |
| B2 | IGRA v2 soundings (NOAA NCEI) | large per-station archives; the risk is silently pulling a derived product rather than the raw sounding |

`ssd.jpl.nasa.gov` already needed `requests` rather than `curl` on this machine
because of an intercepting TLS proxy (`docs/DATA.md`). The same may apply to
these hosts. **Proposal:** the first implementation step for each track is a
loader that fetches, checksums and unit-asserts *one* file, reported
before any modelling, so a bad archive fails in minutes rather than after a
simulation half is built.

### 2.3 Part A's compute budget is the second risk

The outer objective is `differential_evolution` over spline knot values; each
evaluation runs an inner `least_squares` M-estimate per injection per seed.
With `k` knots, a population of `15k`, 100 generations, 20 injections and 3
seeds, that is `15k × 100 × 60` inner solves; for 6 knots, **540 000**. Each
is milliseconds, so it is hours, not days, but it is far beyond anything
currently in the repo (the whole test suite is 11 s).

**Proposal:** cap it explicitly: `maxiter`, `popsize` and the injection count
as named constants with a comment justifying each, a `--quick` path for CI,
and the search cached under `.cache/loss/` keyed by a hash of the injections
and the search configuration, exactly as `.cache/sr/` already is.

### 2.4 A new dependency needs a decision

B1's simulation half wants Mie scattering: `miepython`, or an in-repo series.
The repo's dependency discipline is strict (heavy things are optional extras,
and the committed results record the version that produced them). An in-repo
Mie series is ~40 lines, has no install story, and gets a convergence study on
its truncation anyway, which the spec already requires. **Recommendation:**
write it in `numerics/`, no new dependency. The alternative is
`miepython` as a `[mie]` extra.

### 2.5 An ordering conflict to resolve

You approved Phase 3 as `quantum/helium` + `gravity/pulsar_spindown`, the two
"law is wrong by design" tracks, chosen because the PINN's only decisive win
is on the one existing track whose law is incomplete. Phase 5 adds three more
tracks plus Part A. Five new tracks is more than the repo's headline can
absorb at once, and B1 is *also* a "law is an approximation" track, so it
tests the same mechanism as helium.

Options, in order of preference:

1. **B1 replaces `pulsar_spindown` in Phase 3.** Keep `helium` (cleanest
   wrong-prior case, no signal processing), add `em/rain_attenuation` (same
   mechanism, real instrument noise, and it opens Part A's real-data route).
   Drop `pulsar_spindown` to a stub.
2. Phase 3 as approved, then Phase 5 in full afterwards: five new tracks, the
   longest path.
3. Phase 5 only, treating `helium` and `pulsar_spindown` as stubs.

---

## 3 · Order of work

Following `phase5.md` §1, with the stop points from §5:

| # | step | stops at |
|---|---|---|
| 0 | freeze the Phase 2 default config; record it in `docs/METHOD.md` | Phase 2 deliverable table |
| 1 | write the loss-discovery hypothesis into `docs/METHOD.md` **before** running anything (§2.1) | – |
| 2 | Part A synthetic controls: Gaussian, Laplace, Student-t(2,4), Gaussian+5% outliers | **STOP**: report §2.4's table. If Gaussian and Laplace fail, the search is broken and nothing downstream is trustworthy |
| 3 | B1 loader + Mie simulation half | **STOP**: report the power-law deviation from the Mie integral before any arm runs |
| 4 | B1 arms, six protocol questions, figures, notebook section | – |
| 5 | Part A applied to B1's residuals (§2.6, frozen before reporting splits) | – |
| 6 | B2 `atmos/barometric` | – |
| 7 | B3 `em/maxwell_1d`, also the real test of Phase 2's Fourier features | – |
| 8 | integration: `physprior run em|atmos`, notebooks 04–06, headline, README, `test_claims` | final table |

Stubs recorded and not implemented: `em/gaseous_absorption`,
`atmos/clausius_clapeyron`, `em/two_ray`.

---

## 4 · Repository organisation this implies

`em/` and `atmos/` are new problem families beside `gravity`, `relativity` and
`quantum`, each with the existing layout
(`<simulation>.py`, `<track>.py`, `discovery.py`, `run.py`) and registered in
`physprior.problems.PROBLEMS`. That much is mechanical.

Two structural notes:

- **`PROBLEMS` becomes five families and eight-plus tracks.** The flat
  `results/<problem>/<track>/` layout holds, but `docs/<topic>/README.md` and the
  README's problem table will need to be generated rather than hand-written,
  or they will drift; invariant 1 applied to structure rather than numbers.
- **Part A belongs nowhere in `problems/`.** It is a benchmark-level
  experiment, so it goes under `benchmark/` with results at
  `results/loss_discovery/`, and it gets its own notebook rather than a
  section inside a problem's.

---

## 5 · Decisions required

1. **§2.5 ordering**: which of the three options.
2. **§2.4 Mie**: in-repo series (recommended) or a `miepython` extra.
3. **Approval to start**, which by `phase5.md` §5.1 is required before step 1,
   and which cannot take effect until Phase 2's defaults are frozen.

Nothing in Phase 5 is started until 1–3 are answered.
