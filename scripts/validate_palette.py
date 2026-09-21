#!/usr/bin/env python3
"""Thin wrapper so the validator can be run without installing the package."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from physprior.viz.palette import main

if __name__ == "__main__":
    main()
