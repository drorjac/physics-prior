# Documentation

One folder per physics topic, plus the cross-cutting documents and the
planning record.

## Topics

| Topic | What is in it |
|---|---|
| [**gravity**](gravity/) | Newtonian orbits, the three-body problem, the solar system from JPL initial conditions; Kepler's third law from DE441 |
| [**relativity**](relativity/) | Schwarzschild precession and post-Newtonian inspirals; GW150914 strain and Mercury's ephemeris |
| [**quantum**](quantum/) | the Schrödinger equation, bound states and tunnelling; NIST hydrogen levels and the COBE/FIRAS blackbody |
| [**fields**](fields/) | spatial fields: NOAA station temperatures and the lapse rate over the Alps; a simulated radio field and the path-loss law |
| [**neglected**](neglected/) | the controlled study that cuts across all three: **when** a physics prior helps, on an algebraic law, an ODE and a PDE |

The physics topics carry that problem's simulations, its real-data tracks, its
figures and animations, and the caveats that belong to it. **neglected** is
different in kind: it is the controlled experiment where the answer is known
in advance, and it is the one page to read if you only read one.

## Studies across problems

| Study | What it answers |
|---|---|
| [reconstruction/](reconstruction/) | how the value of a physics prior for reconstructing a field from sparse sensors changes from 1-D to 3-D ([hypothesis, written first](reconstruction/HYPOTHESIS.md)) |
| [lorenz/](lorenz/) | a chaotic inverse problem: a PINN against a tuned black box and classical shooting, under noise and with less data; the PINN's training recipe built up one piece at a time; the butterfly effect |
| [dynamics/](dynamics/) | learning the update rule of an ODE or PDE: what structure buys over long rollouts |
| [optimization/](optimization/) | optimizers, learning rates, loss functions and curvature for physics-constrained models and black boxes; what loss balancing does to the `pinn` arm |
| [theory/symbolic_regression.md](theory/symbolic_regression.md) | how symbolic regression searches, measured with in-repo implementations, PySR and SINDy |
| [theory/packages.md](theory/packages.md) | what `curve_fit`, autograd, Adam and SymPy do underneath, checked against the installed versions |

## Cross-cutting

| Document | What it answers |
|---|---|
| [MISSIONS.md](MISSIONS.md) | every task: goal, conclusion for the PINN, and status, clear results first |
| [CONTRIBUTIONS.md](CONTRIBUTIONS.md) | what the project adds, how new each part is, and how strong the evidence |
| [HYPOTHESES.md](HYPOTHESES.md) | the physical questions the project answers, and what would refute each |
| [DECISIONS.md](DECISIONS.md) | the research decisions taken, and the ones still open |
| [RELATED_WORK.md](RELATED_WORK.md) | the prior work, and what this project adds to it |
| [METHOD.md](METHOD.md) | the design decisions, and the ones made *after* something went wrong |
| [DATA.md](DATA.md) | provenance, units, and the caveats stated up front |
| [TOOLING.md](TOOLING.md) | which package does what, and exactly how a formula comes out of symbolic regression |
| [REPRODUCTION.md](REPRODUCTION.md) | the last full reproduction: what was re-run, what matched, and on what |
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
