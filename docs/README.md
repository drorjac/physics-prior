# Documentation

One folder per physics topic, plus the cross-cutting documents and the
planning record.

## Topics

| Topic | What is in it |
|---|---|
| [**gravity**](gravity/) | Newtonian orbits, the three-body problem, the solar system from JPL initial conditions; Kepler's third law from DE441 |
| [**relativity**](relativity/) | Schwarzschild precession and post-Newtonian inspirals; GW150914 strain and Mercury's ephemeris |
| [**quantum**](quantum/) | the Schrödinger equation, bound states and tunnelling; NIST hydrogen levels and the COBE/FIRAS blackbody |

Each topic page carries that problem's simulations, its real-data tracks, its
figures and animations, and the caveats that belong to it.

## Cross-cutting

| Document | What it answers |
|---|---|
| [METHOD.md](METHOD.md) | the design decisions, and the ones made *after* something went wrong |
| [DATA.md](DATA.md) | provenance, units, and the caveats stated up front |
| [TOOLING.md](TOOLING.md) | which package does what, and exactly how a formula comes out of symbolic regression |
| [RESULTS.md](RESULTS.md) | **generated** from `results/` by `physprior report` — do not edit |

## Planning

| Document | State |
|---|---|
| [plans/PLAN.md](plans/PLAN.md) | the audit, the PINN scorecard, and the Phase 2 outcome |
| [plans/PHASE5_PLAN.md](plans/PHASE5_PLAN.md) | EM propagation, atmosphere and loss discovery — **at its approval gate** |
| [plans/PHASE6_PLAN.md](plans/PHASE6_PLAN.md) | quantum interference tracks — **at its approval gate** |
| [plans/phase5-spec.md](plans/phase5-spec.md) | the Phase 5 specification as written |

---

Numbers in these pages come from `results/` and are regenerated rather than
typed; the few quoted in prose are re-derived by
[`tests/test_claims.py`](../tests/test_claims.py), which fails if they drift.
