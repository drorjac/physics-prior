"""Allow `python -m physprior` as well as the `physprior` console script."""

from .cli import main

raise SystemExit(main())
