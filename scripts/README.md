# scripts

| Script | What it does |
|---|---|
| `regenerate.sh` | the full pipeline: run every problem, write figures, regenerate the docs and README tables, execute the notebooks, then run the tests |
| `validate_palette.py` | run the figure-palette validator without installing the package (it is also `python -m physprior.viz.palette`) |

Both assume the package is installed (`pip install -e '.[all]'`).
