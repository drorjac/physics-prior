"""Notebook cells and the shared set-up, for every tutorial generator.

Kept apart from `tutorials.py` so the topic modules (T8 onwards) can use the
same helpers without importing the module that registers them.
"""

from __future__ import annotations

import nbformat as nbf

SETUP = """import warnings
warnings.filterwarnings("ignore")
import numpy as np, torch, matplotlib.pyplot as plt
from physprior.viz import plots as P
P.use_style()
torch.set_default_dtype(torch.float64)
print("torch", torch.__version__)
"""


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def _nb(cells, title):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {
            "display_name": "physprior",
            "language": "python",
            "name": "physprior",
        },
        "language_info": {"name": "python"},
        "title": title,
    }
    return nb
