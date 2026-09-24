"""Regenerate the neglected-terms study: its tables and its figures.

`physprior neglected [--quick]`

This exists because the study's outputs were **orphaned**. Five CSVs under
`results/` and fourteen figures under `figures/neglected/` were committed,
and nothing in the repository reproduced them -- `scripts/regenerate.sh` ran
every problem, wrote every other figure and rebuilt the docs, and skipped
this study entirely. A committed artefact that no command regenerates is the
exact drift the project's invariants exist to prevent: it cannot be checked,
and it silently stops matching the code that supposedly produced it.

The study asks one question on three rungs of the same ladder:

    algebraic   y = GM/r^2 + missing(r)
    ODE         theta'' = -omega^2 sin(theta) + missing(theta, theta')
    PDE         u_t = alpha u_xx + missing(u, u_x)

and the answer is the same on all three: **a physics prior helps when the
missing piece is distinguishable from the law, not merely when the law is
incomplete.** A missing term shaped like the law is absorbed into the law's
own constant, which then comes back wrong while the fit looks fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.config import get_settings, short_path
from physprior.viz import plots as P

TRACK = "neglected"


def _write(df: pd.DataFrame, name: str) -> pd.DataFrame:
    path = get_settings().results_dir / f"neglected_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {short_path(path)}")
    return df


def _write_tune(df: pd.DataFrame, name: str) -> None:
    d = get_settings().results_dir / TRACK
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"tune_{name}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {short_path(path)}")


def _save(fig, name: str) -> None:
    out = P.save(fig, TRACK, name)
    print(f"  wrote {short_path(out)}")


def algebraic(quick: bool = False) -> None:
    """Rung 1: the law is an algebraic relation."""
    import physprior.benchmark.neglected as N

    epochs = 600 if quick else 3000
    kw: dict = {"epochs": epochs}
    if quick:
        kw["seeds"] = (11,)
    # The grids are the committed ones; see results/neglected_*.csv.
    eps_grid = [0.0, 0.2] if quick else [0.0, 0.1, 0.2, 0.4, 0.8]
    noise_grid = [0.0, 0.1] if quick else [0.0, 0.02, 0.05, 0.1, 0.2]
    budget_grid = [10, 160] if quick else [10, 20, 40, 80, 160]

    print("rung 1: algebraic")
    eps = _write(N.sweep(eps_grid, "eps", noise=0.02, **kw), "eps")
    _save(
        P.fig_neglected_sweep(
            eps,
            "eps",
            "size of the term left out of the law  (eps)",
            "The physics prior pays as soon as the law is incomplete",
        ),
        "01_eps",
    )
    noise = _write(N.sweep(noise_grid, "noise", eps=0.2, **kw), "noise")
    _save(
        P.fig_neglected_sweep(
            noise,
            "noise",
            "measurement noise (fraction of signal spread)",
            "...and stops paying when the correction starts fitting noise",
        ),
        "02_noise",
    )
    budget = _write(
        N.sweep(budget_grid, "n_train", eps=0.2, noise=0.02, **kw), "budget"
    )
    _save(
        P.fig_neglected_sweep(
            budget,
            "n_train",
            "training points",
            "The black box needs data to reach a prior it never beats",
            logx=True,
        ),
        "03_budget",
    )

    # the DEGENERATE control: a missing term shaped like the law itself
    deg = N.sweep(eps_grid, "eps", noise=0.02, shape="power", **kw)
    _save(
        P.fig_neglected_sweep(
            deg,
            "eps",
            "size of the term left out of the law  (eps)",
            "When the missing term LOOKS like the law, it is absorbed into GM",
            metric="gm_error_pct",
        ),
        "06_degenerate",
    )


def ode(quick: bool = False) -> None:
    """Rung 2: the law is an ordinary differential equation."""
    import physprior.benchmark.neglected as N

    epochs = 600 if quick else 3000
    seeds = (11,) if quick else N.REPORT_SEEDS
    print("rung 2: ODE")
    rows = []
    amps = [10, 60] if quick else [10, 30, 60, 90, 120]
    for shape in ("damping", "anharmonic"):
        for amp in amps:
            for sd in seeds:
                for res in N.run_ode(
                    amplitude=amp, shape=shape, seed=sd, epochs=epochs
                ):
                    rows.append(
                        {
                            "shape": shape,
                            "amplitude_deg": amp,
                            "seed": sd,
                            "arm": res.arm,
                            "nrmse": res.nrmse_in,
                            "omega_error_pct": res.gm_error_pct,
                            "correction_error": res.correction_error,
                        }
                    )
            print(f"  {shape} amplitude={amp} done", flush=True)
    df = _write(pd.DataFrame(rows), "ode")
    for shape, name, title in (
        (
            "damping",
            "10_ode_sweep_damping",
            "A missing FORCE the law cannot imitate: the PINN wins",
        ),
        (
            "anharmonic",
            "11_ode_sweep_anharmonic",
            "A missing force shaped like the law: absorbed into omega",
        ),
    ):
        _save(
            P.fig_neglected_sweep(
                df[df["shape"] == shape],
                "amplitude_deg",
                "initial amplitude  [degrees]",
                title,
                metric="nrmse",
            ),
            name,
        )


def pde(quick: bool = False) -> None:
    """Rung 3: the law is a partial differential equation.

    The `pinn` arm here does NOT converge and is recorded that way; see
    `pde_pinn_converged` and the post-mortem in docs/METHOD.md. The rung's
    conclusion rests on the `physics` arm, whose result is a closed-form
    identity and does not depend on any network.
    """
    import physprior.benchmark.neglected as N

    epochs = 600 if quick else 3000
    seeds = (11,) if quick else N.REPORT_SEEDS
    print("rung 3: PDE")
    rows = []
    for shape in ("diffusive", "advective"):
        for eps in [0.0, 0.3] if quick else [0.0, 0.15, 0.3]:
            for sd in seeds:
                for res in N.run_pde(eps=eps, shape=shape, seed=sd, epochs=epochs):
                    rows.append(
                        {
                            "shape": shape,
                            "eps": eps,
                            "seed": sd,
                            "arm": res.arm,
                            "nrmse": res.nrmse_in,
                            "alpha_error_pct": res.gm_error_pct,
                        }
                    )
            print(f"  {shape} eps={eps} done", flush=True)
    df = _write(pd.DataFrame(rows), "pde")

    for shape, name in (
        ("diffusive", "12_pde_diffusive"),
        ("advective", "12_pde_advective"),
    ):
        sub = df[(df["shape"] == shape) & (df["arm"] == "physics")]
        alpha_err = float(sub["alpha_error_pct"].median())
        sys_ = N.NeglectedPDE(eps=0.3, noise=0.0, shape=shape)
        _save(P.fig_pde_field(sys_, sys_.alpha * (1 + alpha_err / 100)), name)


def detail(quick: bool = False) -> None:
    """The single-run figures: what one fit actually learned.

    The sweeps say which arm wins. These say WHY, and they are the point of
    the study: a correction drawn against the term it was never shown, and a
    loss split into its parts so that "the loss went down" and "the constant
    converged" can be seen to be different events.
    """
    import physprior.benchmark.neglected as N

    epochs = 600 if quick else 3000
    print("detail: what a single fit learned")

    # --- rung 1, the distinguishable case -------------------------------
    sys_ = N.NeglectedSystem(eps=0.2, noise=0.02, shape="bump")
    r, y, _ = sys_.sample(40, seed=11)
    _, _, _, correction, hist = N._fit_pinn(
        sys_, r, y, w_phys=1.0, epochs=epochs, seed=11
    )
    _save(
        P.fig_learned_correction(
            sys_,
            correction,
            title="The correction the PINN learned, against the term it never saw",
        ),
        "04_learned_correction",
    )
    _save(
        P.fig_learning_curves(
            hist,
            published=N.GM_TRUE,
            title="The loss going down and the constant converging are different events",
        ),
        "05_learning_curves",
    )

    # --- rung 2, both shapes --------------------------------------------
    for shape, amp, name, title in (
        (
            "damping",
            60.0,
            "07_ode_damping",
            "A missing force the harmonic law cannot imitate",
        ),
        (
            "anharmonic",
            60.0,
            "09_ode_anharmonic",
            "A missing force shaped like the law: absorbed into omega",
        ),
    ):
        sys_ = N.NeglectedODE(amplitude=amp, noise=0.02, shape=shape)
        t, theta = sys_.sample(60, seed=11)
        predict, _, _, force, hist = N._ode_fit_pinn(
            sys_, t, theta, w_phys=1e-3, epochs=epochs, seed=11
        )
        phys_predict, _ = N._ode_fit_physics(sys_, t, theta)
        _save(
            P.fig_learned_force(
                sys_, force, predict=predict, physics=phys_predict, title=title
            ),
            name,
        )
        if shape == "damping":
            _save(
                P.fig_learned_force(sys_, force, title="The learned force alone"),
                "07_ode_force",
            )
            _save(
                P.fig_learning_curves(
                    hist, title="Training the residual PINN on the pendulum"
                ),
                "08_ode_learning",
            )


def derivative_accuracy(quick: bool = False) -> None:
    """Why the PDE PINN's constant is wrong while its field looks right.

    The most transferable figure in the study, and the one the failure of
    rung 3 turned out to be about.
    """
    import physprior.benchmark.neglected as N

    print("diagnostic: fitting a field vs differentiating it")
    out = N.derivative_accuracy_study(
        n=200 if quick else 500,
        epochs=600 if quick else 6000,
        weights=(0.0, 3e-2) if quick else (0.0, 3e-3, 1e-2, 3e-2),
    )
    print(
        f"  data loss {out['data_loss']:.2e}; mean |u_xx| "
        f"{out['curvature'][0]:.3f} against an exact {out['curvature_exact']:.3f}"
    )
    _save(
        P.fig_derivative_accuracy(
            out["x"],
            out["u_exact"],
            out["u_net"],
            out["uxx_exact"],
            out["uxx_net"],
            (out["weights"], out["alphas"]),
            out["alpha_true"],
            title="A small data loss does not buy a right constant",
        ),
        "13_derivative_accuracy",
    )


def tune(quick: bool = False) -> None:
    """The two design sweeps behind the PDE rung, on the TUNING seeds.

    Both answer "why is this configured the way it is", so both run on
    3/7/19 and neither may be read off a reporting seed (invariant 6). They
    are regenerated here rather than pasted, because a hand-typed table is
    exactly what invariant 1 forbids.

    Run at `eps = 0`, where the modelled law is exactly right and `alpha`
    must come back 0.05, so any deviation is the method's own.
    """
    import physprior.benchmark.neglected as N

    epochs = 600 if quick else 4000
    seeds = (3,) if quick else (3, 7, 19)
    true = 0.05
    print("tuning (seeds 3/7/19, eps = 0)")

    rows: list[dict] = []
    for w in (0.0, 0.03) if quick else (0.0, 3e-3, 1e-2, 3e-2, 1e-1):
        a = []
        for sd in seeds:
            sys_ = N.NeglectedPDE(eps=0.0, noise=0.02, shape="diffusive")
            xs, ts, us = sys_.sample(500, seed=sd)
            _, al, _, _ = N._pde_fit_pinn(
                sys_, xs, ts, us, epochs=epochs, seed=sd, w_smooth=w
            )
            a.append(al)
        arr = np.asarray(a)
        rows.append(
            dict(
                w_smooth=w,
                alpha_mean=arr.mean(),
                alpha_std=arr.std(),
                err_pct=abs(arr.mean() / true - 1) * 100,
            )
        )
        print(
            f"  w_smooth={w:<7g} alpha {arr.mean():.5f} "
            f"err {rows[-1]['err_pct']:5.1f}%",
            flush=True,
        )
    _write_tune(pd.DataFrame(rows), "pde_smooth")

    # Is the FREE correction C(u, u_x) what destroys identifiability? For any
    # alpha there is a C satisfying u_t - alpha u_xx - C = 0 exactly, so only
    # w_phys*mean(C^2) pins alpha down. At eps = 0 the true C is zero, so
    # removing it costs nothing and the test is clean.
    ablation: list[dict] = []
    settings = [
        ("C free, w_phys=1e-3", dict(use_correction=True, w_phys=1e-3)),
        ("C penalised, w_phys=1e0", dict(use_correction=True, w_phys=1e0)),
        ("C penalised, w_phys=1e3", dict(use_correction=True, w_phys=1e3)),
        ("C removed (c == 0)", dict(use_correction=False, w_phys=0.0)),
    ]
    for label, kw in settings[:2] if quick else settings:
        a = []
        for sd in seeds:
            sys_ = N.NeglectedPDE(eps=0.0, noise=0.02, shape="diffusive")
            xs, ts, us = sys_.sample(500, seed=sd)
            _, al, _, _ = N._pde_fit_pinn(
                sys_, xs, ts, us, epochs=epochs, seed=sd, **kw
            )
            a.append(al)
        arr = np.asarray(a)
        ablation.append(
            {
                "setting": label,
                "alpha_mean": float(arr.mean()),
                "alpha_std": float(arr.std()),
                "err_pct": float(abs(arr.mean() / true - 1) * 100),
            }
        )
        print(
            f"  {label:<26s} alpha {arr.mean():.5f} err {ablation[-1]['err_pct']:5.1f}%",
            flush=True,
        )
    _write_tune(pd.DataFrame(ablation), "pde_correction")


def main(quick: bool = False, only: str | None = None) -> None:
    stages = {
        "algebraic": algebraic,
        "ode": ode,
        "pde": pde,
        "detail": detail,
        "derivative": derivative_accuracy,
        "tune": tune,
    }
    if only:
        stages = {k: v for k, v in stages.items() if only in k}
        if not stages:
            raise SystemExit(f"no stage matches {only!r}")
    print(f"neglected-terms study{' (quick)' if quick else ''}")
    for fn in stages.values():
        fn(quick=quick)
    print("done")


if __name__ == "__main__":  # pragma: no cover
    main()
