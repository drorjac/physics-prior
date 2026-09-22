# scripts

| Script | What it does |
|---|---|
| `regenerate.sh` | the full pipeline: run every problem, write figures, regenerate the docs and README tables, execute the notebooks, then run the tests |
| `validate_palette.py` | run the figure-palette validator without installing the package (it is also `python -m physprior.viz.palette`) |
| `audit_pinn.py` | where the `pinn` arm wins, loses and ties against the best non-oracle arm, with a gap smaller than the seed spread reported as a tie (the tables in `docs/PLAN.md` §2) |

Both assume the package is installed (`pip install -e '.[all]'`).
