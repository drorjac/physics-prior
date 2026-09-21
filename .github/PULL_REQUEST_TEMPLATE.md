## What this changes

<!-- One or two sentences. The diff says what; say why. -->

## Checklist

- [ ] `make lint` passes (ruff, ruff format, mypy)
- [ ] `make test` passes
- [ ] If a **result** changed: `physprior report` re-run, `CHANGELOG.md` updated,
      and `tests/test_claims.py` still passes
- [ ] If a **numerical method** was added or changed: a convergence study goes
      with it (see rule 5 in `CONTRIBUTING.md`)
- [ ] If a **threshold or hyperparameter** was chosen: it was selected on
      tuning seeds or injected signals, never on the reported data

## Results affected

<!-- Which files under results/ change, and by how much. "None" is a fine answer. -->
