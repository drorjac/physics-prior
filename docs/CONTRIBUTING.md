# Contributing

## Getting set up

```bash
git clone https://github.com/drorjac/physics-prior
cd physics-prior
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pre-commit install
```

`torch` and `pysr` are optional extras. Everything except the `nn`/`pinn` and
`sr` arms runs without them, and the test suite marks what needs each.

```bash
make test          # fast: no network, no Julia
make test-all      # everything, including injection studies
make lint          # ruff + ruff format --check + mypy
```

## The rules this project holds itself to

These are not style preferences. Each one is here because breaking it produced
a wrong physical result at some point, and the story is in `docs/METHOD.md`.

1. **No number in the prose is typed by hand without a test behind it.**
   Tables in the README and `docs/RESULTS.md` are generated from `results/` by
   `physprior report`. The few numbers quoted in narrative are recomputed by
   `tests/test_claims.py`, which fails if they drift.

2. **Units are asserted, with `require`, never `assert`.** `assert` disappears
   under `python -O`, and a guard that vanishes under optimisation is worse
   than no guard. Every loader states the range it promises.

3. **Physical constants carry their source.** See `physprior/constants.py`;
   add the citation in the comment, not just the value.

4. **A parameter pinned to its bound has not converged**, whatever the
   optimiser reports. Check and mark it.

5. **A numerical result that is still moving with step size is not a result.**
   If you add a derivative, an integrator or a solver, add the convergence
   study with it. This project once reported a 56σ refutation of general
   relativity that was entirely finite-difference truncation error.

6. **Tune on seeds 3 / 7 / 19, report on 11 / 23 / 42.** Never select a
   hyperparameter, a threshold or a design choice by looking at a reported
   seed, or at the real data. Signal-processing choices are calibrated on
   injected signals with known answers.

7. **Negative results are results.** Symbolic regression fails to find
   Planck's law here; that failure is reported at the same size as the
   successes, with a control that isolates its cause. Do not spin.

8. **The oracle is not a competitor.** It is the published law with published
   constants. An arm that beats it in-sample is fitting noise; an arm that
   beats it out-of-sample needs an explanation, and `physprior report` will
   ask for one.

## Adding a physics problem

A problem is a folder under `src/physprior/problems/` with, at minimum:

```text
<name>/
  __init__.py
  <simulation>.py   where the law is exactly what you put in
  <track>.py        problem() -> (Problem, meta); run() for the sweeps
  discovery.py      recover the law from the simulation
  run.py            run(quick: bool) -> dict
```

Register it in `physprior.problems.PROBLEMS`. The benchmark protocol is shared
(`physprior.benchmark.protocol`); do not reimplement the sweeps.

If it needs a new dataset, add a loader under `physprior/data/sources/` and
register it in `physprior/data/registry.py`. Loaders assert their units and
record provenance; they know nothing about fitting.

## Commit and PR conventions

- One logical change per commit; the message says *why*, the diff says what.
- CI must pass: ruff, ruff format, mypy, and the offline test subset.
- If a result changes, say so in `CHANGELOG.md` and re-run `physprior report`.
