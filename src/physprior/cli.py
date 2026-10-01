"""Command-line interface.

    physprior run [gravity|relativity|quantum|all] [--quick]
    physprior report [--check]
    physprior verify CANDIDATE [--reference DIR]
    physprior figures
    physprior notebooks [--execute] [NAME]
    physprior data list
    physprior data fetch NAME
    physprior neglected [STAGE] [--quick]
    physprior theory [--quick] [--parts P ...]
    physprior optim [--quick]
    physprior reconstruct [--quick] [--workers N] [--doc-only]
    physprior dynamics [--quick] [--doc-only]
    physprior lorenz [--quick] [--workers N] [--doc-only] [--no-gif]
    physprior budget-weight [TRACK]
    physprior summary [--execute]
    physprior summary-md [--check]
    physprior info

This is the only module that configures logging or prints; everything under it
emits log records and returns values.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence

from physprior import __version__
from physprior.config import get_settings
from physprior.exceptions import PhysPriorError
from physprior.logging import configure, get_logger
from physprior.problems import PROBLEMS

log = get_logger(__name__)

_ALIASES = {"g": "gravity", "r": "relativity", "q": "quantum"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="physprior",
        description="What does a physics prior buy you? "
        "Benchmarks on real physics data and on simulations.",
    )
    parser.add_argument(
        "--version", action="version", version=f"physprior {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="debug-level logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="warnings and errors only"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a physics problem end to end")
    run.add_argument(
        "problem",
        nargs="?",
        default="all",
        help=f"{' | '.join(PROBLEMS)} | all  (default: all)",
    )
    run.add_argument(
        "--quick", action="store_true", help="short sweeps, for a smoke test"
    )

    rep = sub.add_parser("report", help="results/ -> headline.json, docs, README")
    rep.add_argument(
        "--check",
        action="store_true",
        help="write nothing; exit 1 if a generated file is out of date",
    )

    ver = sub.add_parser(
        "verify",
        help="compare a regenerated results directory with the committed one",
    )
    ver.add_argument("candidate", help="the regenerated results directory")
    ver.add_argument(
        "--reference",
        default=None,
        help="the directory to compare against (default: the resolved results dir)",
    )
    ver.add_argument("--rtol", type=float, default=None, help="relative tolerance")
    ver.add_argument("--atol", type=float, default=None, help="absolute tolerance")
    sub.add_parser("figures", help="write figures/ from results/")

    nb = sub.add_parser("notebooks", help="build (and optionally run) notebooks")
    nb.add_argument(
        "name",
        nargs="?",
        default=None,
        help="only notebooks whose filename contains this",
    )
    nb.add_argument(
        "--execute", action="store_true", help="execute them after building"
    )

    tut = sub.add_parser(
        "tutorials", help="build (and optionally run) the step-by-step PINN course"
    )
    tut.add_argument(
        "name", nargs="?", default=None, help="only tutorials matching this"
    )
    tut.add_argument(
        "--execute", action="store_true", help="execute them after building"
    )

    data = sub.add_parser("data", help="inspect and fetch datasets")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    data_sub.add_parser("list", help="list the known datasets")
    fetch = data_sub.add_parser("fetch", help="download one dataset now")
    fetch.add_argument("name")

    tune = sub.add_parser(
        "tune",
        help="choose the pinn arm's defaults on the TUNING seeds (3/7/19)",
    )
    tune.add_argument(
        "track",
        nargs="?",
        default=None,
        help="one track, e.g. gravity/kepler (default: all four)",
    )
    tune.add_argument(
        "--quick", action="store_true", help="shorter training, for a smoke test"
    )
    tune.add_argument(
        "--what",
        choices=("ablation", "w_phys", "both"),
        default="both",
        help="which study to run (default: both)",
    )

    neg = sub.add_parser(
        "neglected",
        help="regenerate the neglected-terms study (tables + figures)",
    )
    neg.add_argument(
        "stage",
        nargs="?",
        default=None,
        help="algebraic | ode | pde | detail | derivative | tune | consistency (default: all)",
    )
    neg.add_argument(
        "--quick", action="store_true", help="short training, for a smoke test"
    )

    th = sub.add_parser(
        "theory",
        help="symbolic-regression theory studies: search space, GP, SINDy, packages",
    )
    th.add_argument("--quick", action="store_true", help="short runs, for a smoke test")
    th.add_argument(
        "--parts",
        nargs="*",
        default=None,
        help="growth tune noise budget pysr vocabulary pareto sindy packages",
    )

    opt = sub.add_parser(
        "optim",
        help="optimizers, loss functions, curvature and the balance diagnostic",
    )
    opt.add_argument(
        "--quick", action="store_true", help="short runs, for a smoke test"
    )

    rec = sub.add_parser(
        "reconstruct",
        help="sparse-sensor field reconstruction in 1-D, 2-D and 3-D",
    )
    rec.add_argument(
        "--quick", action="store_true", help="short runs, for a smoke test"
    )
    rec.add_argument("--workers", type=int, default=None, help="parallel processes")
    rec.add_argument(
        "--doc-only", action="store_true", help="re-render the page from results/"
    )

    dyn = sub.add_parser(
        "dynamics",
        help="learning the update rule of ODEs and PDEs: learned steppers vs structure",
    )
    dyn.add_argument(
        "--quick", action="store_true", help="short runs, for a smoke test"
    )
    dyn.add_argument(
        "--doc-only", action="store_true", help="redraw figures and page from results/"
    )

    lz = sub.add_parser(
        "lorenz",
        help="Lorenz-63: PINN vs black box vs shooting, noise, budget, "
        "optimisation ladder, the butterfly effect",
    )
    lz.add_argument("--quick", action="store_true", help="short runs, for a smoke test")
    lz.add_argument("--workers", type=int, default=None, help="parallel processes")
    lz.add_argument(
        "--doc-only", action="store_true", help="redraw figures and page from results/"
    )
    lz.add_argument(
        "--no-gif", action="store_true", help="skip the butterfly animation"
    )

    bw = sub.add_parser(
        "budget-weight",
        help="H7: the pinn arm's physics weight chosen per training-set size",
    )
    bw.add_argument("track", nargs="?", default=None, help="one track (default: all)")

    smd = sub.add_parser("summary-md", help="write docs/SUMMARY.md from results/")
    smd.add_argument(
        "--check", action="store_true", help="write nothing; exit 1 if it is stale"
    )

    sm = sub.add_parser("summary", help="build the one-notebook tour of the results")
    sm.add_argument("--execute", action="store_true", help="execute it after building")

    sub.add_parser("info", help="show resolved paths and configuration")
    return parser


def _tune(track: str | None, quick: bool, what: str) -> int:
    """Everything here runs on the tuning seeds and writes under
    results/<track>/tune/, so no default is ever chosen by looking at a
    reported number (invariant 6)."""
    from physprior.benchmark import ablation

    tracks = [track] if track else None
    if what in ("ablation", "both"):
        log.info("ablating the pinn options on the tuning seeds")
        df = ablation.run(tracks=tracks, quick=quick)
        ablation.save(df, "ablation")
        summary = ablation.summarise(df)
        ablation.save(summary, "ablation_summary")
        print("\n" + summary.to_string(index=False))
        print("\n" + ablation.decide(summary).to_string(index=False))
    if what in ("w_phys", "both"):
        log.info("sweeping w_phys on the tuning seeds")
        w = ablation.tune_physics_weight(tracks=tracks, quick=quick)
        ablation.save(w, "sweep_physics_weight")
    return 0


def _report(check: bool) -> int:
    from physprior.config import short_path
    from physprior.reporting import report

    if not check:
        report.build()
        return 0
    stale = report.check()
    for path in stale:
        log.error("out of date: %s -- run `physprior report`", short_path(path))
    if not stale:
        log.info("the generated tables match results/")
    return 1 if stale else 0


def _verify(args: argparse.Namespace) -> int:
    from pathlib import Path

    from physprior.reporting import verify

    reference = Path(args.reference) if args.reference else get_settings().results_dir
    candidate = Path(args.candidate)
    for d in (reference, candidate):
        if not d.is_dir():
            log.error("not a directory: %s", d)
            return 2
    results = verify.compare_dirs(
        reference,
        candidate,
        rtol=verify.RTOL if args.rtol is None else args.rtol,
        atol=verify.ATOL if args.atol is None else args.atol,
    )
    for r in results:
        if r.status != "identical":
            print(f"  {r.status:16s} {r.path}  {r.detail}".rstrip())
    print(verify.summary(results))
    return 1 if any(r.failed for r in results) else 0


def _record_environment() -> None:
    import json

    from physprior.environment import snapshot

    path = get_settings().results_dir / "environment.json"
    path.write_text(json.dumps(snapshot(), indent=2, sort_keys=True) + "\n")


def _run(problem: str, quick: bool) -> int:
    problem = _ALIASES.get(problem, problem)
    names = list(PROBLEMS) if problem == "all" else [problem]
    for name in names:
        if name not in PROBLEMS:
            log.error(
                "unknown problem %r; choose from %s or 'all'", name, list(PROBLEMS)
            )
            return 2
    _record_environment()
    for name in names:
        module = __import__(f"physprior.problems.{name}.run", fromlist=["run"])
        started = time.monotonic()
        log.info("running problem %s%s", name, " (quick)" if quick else "")
        module.run(quick=quick)
        log.info("problem %s finished in %.0f s", name, time.monotonic() - started)
    return 0


def _data(args: argparse.Namespace) -> int:
    from physprior.data.registry import DATASETS, load

    if args.data_command == "list":
        for name, (module, attr) in sorted(DATASETS.items()):
            print(f"{name:20s}  {module}.{attr}")
        return 0
    dataset = load(args.name)
    size = len(dataset) if hasattr(dataset, "__len__") else "?"
    log.info("fetched %s (%s records)", args.name, size)
    return 0


def _info() -> int:
    settings = get_settings()
    print(f"physprior {__version__}")
    for label, value in (
        ("root", settings.root),
        ("raw data", settings.raw_dir),
        ("results", settings.results_dir),
        ("figures", settings.figures_dir),
        ("cache", settings.cache_dir),
        ("notebooks", settings.notebooks_dir),
        ("offline", settings.offline),
    ):
        print(f"  {label:10s} {value}")
    try:
        from physprior.methods import device
    except ImportError:  # torch is an optional extra
        print("  torch      not installed ([nn] extra)")
        return 0
    print(f"  torch      {device.describe()}")
    if device.available()["cuda"] or device.available()["mps"]:
        print("             (the GPU paths are UNVERIFIED -- see methods/device.py)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    configure("DEBUG" if args.verbose else "WARNING" if args.quiet else "INFO")
    get_settings().ensure_dirs()
    try:
        if args.command == "run":
            return _run(args.problem, args.quick)
        if args.command == "report":
            return _report(args.check)
        if args.command == "verify":
            return _verify(args)
        if args.command == "figures":
            from physprior.reporting.figures import main as figures_main

            figures_main()
            return 0
        if args.command == "notebooks":
            from physprior.reporting.notebooks import build as build_notebooks

            build_notebooks(execute=args.execute, only=args.name)
            return 0
        if args.command == "tutorials":
            from physprior.reporting.tutorials import build as build_tutorials

            build_tutorials(execute=args.execute, only=args.name)
            return 0
        if args.command == "tune":
            return _tune(args.track, args.quick, args.what)
        if args.command == "data":
            return _data(args)
        if args.command == "neglected":
            from physprior.reporting.neglected_study import main as neglected_main

            _record_environment()
            neglected_main(quick=args.quick, only=args.stage)
            return 0
        if args.command == "theory":
            from physprior.symbolic import studies

            studies.run(quick=args.quick, parts=args.parts)
            return 0
        if args.command == "optim":
            from physprior.optim import report as optim_report

            optim_report.run_all(quick=args.quick)
            optim_report.make_figures()
            optim_report.render_doc()
            return 0
        if args.command == "reconstruct":
            if args.doc_only:
                from physprior.reconstruction.doc import render_doc

                render_doc()
            else:
                from physprior.reconstruction.study import run as reconstruct

                reconstruct(quick=args.quick, workers=args.workers)
            return 0
        if args.command == "dynamics":
            from physprior.dynamics import doc as dynamics_doc
            from physprior.dynamics import figures as dynamics_figures

            if args.doc_only:
                dynamics_figures.regenerate()
            else:
                from physprior.dynamics.study import run as run_dynamics

                run_dynamics(quick=args.quick)
            dynamics_doc.render_doc()
            return 0
        if args.command == "lorenz":
            from physprior.lorenz import doc as lorenz_doc
            from physprior.lorenz import figures as lorenz_figures

            if not args.doc_only:
                from dataclasses import replace

                from physprior.lorenz.study import Options
                from physprior.lorenz.study import run as run_lorenz

                o = Options.quick_options() if args.quick else Options()
                if args.workers:
                    o = replace(o, workers=args.workers)
                run_lorenz(opts=o)
            lorenz_figures.regenerate(gif=not args.no_gif)
            lorenz_doc.render_doc()
            return 0
        if args.command == "budget-weight":
            from physprior.benchmark import budget_weight

            budget_weight.run(only=args.track)
            v = budget_weight.verdicts()
            print({k: v[k] for k in v if k != "table"})
            return 0
        if args.command == "summary-md":
            from physprior.reporting import project_summary

            if args.check:
                ok = project_summary.check()
                print("SUMMARY.md is up to date" if ok else "SUMMARY.md is stale")
                return 0 if ok else 1
            project_summary.render()
            return 0
        if args.command == "summary":
            from physprior.reporting.summary import build as build_summary

            build_summary(execute=args.execute)
            return 0
        if args.command == "info":
            return _info()
    except PhysPriorError as exc:
        # An expected failure: report it as a message, not a traceback.
        log.error("%s: %s", type(exc).__name__, exc)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        log.warning("interrupted")
        return 130
    log.error("unhandled command %r", args.command)
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
