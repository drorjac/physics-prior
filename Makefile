# physprior -- development tasks.
#
# The package is installed into the active environment (`pip install -e '.[dev]'`),
# so PYTHON is just whatever python is on PATH. Override it if you must:
#     make test PYTHON=/path/to/python
PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
export OMP_NUM_THREADS ?= 2

.PHONY: help install dev all run quick gravity relativity quantum report figures \
        notebooks test test-all test-offline lint format typecheck hooks kernel \
        palette build clean clean-figs reproduce reproduce-quick check-report

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

install:  ## runtime install
	$(PIP) install -e .

dev:  ## development install + git hooks
	$(PIP) install -e '.[dev]'
	pre-commit install

all: run report figures notebooks  ## the whole pipeline

run:  ## run every physics problem -> results/ + figures/
	physprior run all

quick:  ## short sweeps, for a smoke test
	physprior run all --quick

gravity:  ## run one problem
	physprior run gravity
relativity:
	physprior run relativity
quantum:
	physprior run quantum

report:  ## results/ -> headline.json, docs/RESULTS.md, README section
	physprior report

figures:  ## write figures/ from results/
	physprior figures

notebooks:  ## build and execute the notebooks
	physprior notebooks --execute

# The reproduction contract. `reproduce` re-runs the pipeline into a scratch
# directory -- nothing committed is touched -- and then compares every number
# with the committed results/. Wall-clock seconds and environment.json are
# expected to move; anything else that moves is a finding.
REPRO ?= build/reproduce

reproduce:  ## re-run everything into build/reproduce and compare with results/ (hours)
	rm -rf $(REPRO)
	PHYSPRIOR_RESULTS_DIR=$(REPRO)/results PHYSPRIOR_FIGURES_DIR=$(REPRO)/figures \
	  physprior run all
	PHYSPRIOR_RESULTS_DIR=$(REPRO)/results PHYSPRIOR_FIGURES_DIR=$(REPRO)/figures \
	  physprior neglected
	physprior verify $(REPRO)/results
	physprior report --check

reproduce-quick:  ## the same pipeline with short sweeps: exercises the plumbing, will not match
	rm -rf $(REPRO)-quick
	PHYSPRIOR_RESULTS_DIR=$(REPRO)-quick/results PHYSPRIOR_FIGURES_DIR=$(REPRO)-quick/figures \
	  physprior run all --quick
	-physprior verify $(REPRO)-quick/results

check-report:  ## the generated README/RESULTS tables match results/ (seconds)
	physprior report --check

test:  ## fast tests: no network, no Julia
	$(PYTHON) -m pytest -m "not slow and not network and not sr"

test-offline:  ## exactly what CI runs
	PHYSPRIOR_OFFLINE=1 PHYSPRIOR_NO_JULIA=1 \
	  $(PYTHON) -m pytest -m "not slow and not network and not sr"

test-all:  ## everything, including injection studies and network fetches
	$(PYTHON) -m pytest

lint:  ## ruff + format check + mypy
	$(PYTHON) -m ruff check src tests tools
	$(PYTHON) -m ruff format --check src tests
	$(PYTHON) -m mypy

format:  ## apply ruff fixes and formatting
	$(PYTHON) -m ruff check --fix src tests tools
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy

hooks:  ## run pre-commit on everything
	pre-commit run --all-files

kernel:  ## register the Jupyter kernel the notebooks expect
	$(PYTHON) -m ipykernel install --user --name physprior --display-name physprior

palette:  ## re-validate the figure palette
	$(PYTHON) -m physprior.viz.palette "#2a78d6,#eb6834,#1baf7a,#4a3aa7" light all

build:  ## build sdist + wheel and check the metadata
	$(PYTHON) -m build
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf build dist src/*.egg-info .pytest_cache .ruff_cache .mypy_cache

clean-figs:
	rm -rf figures/*
