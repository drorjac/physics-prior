"""Write the figures to `figures/`.

`physprior figures`

The notebooks draw the same figures inline; this writes them as files so the
README and a quick browse do not require running a kernel.
"""

from __future__ import annotations

import numpy as np

from physprior.benchmark.protocol import fit_arm
from physprior.config import get_settings
from physprior.io import load_json, load_table
from physprior.viz import plots as P

TRACKS = {
    "quantum/hydrogen": dict(
        module="physprior.problems.quantum.hydrogen",
        logx=False,
        logy=False,
        param="R",
        published="bohr_rydberg_H_icm",
        extrap="Train on n <= 10, predict n = 11..40",
    ),
    "quantum/cmb": dict(
        module="physprior.problems.quantum.cmb",
        logx=True,
        logy=True,
        param="T",
        published="published_T_K",
        extrap="Train on Rayleigh-Jeans, predict the Wien tail",
    ),
    "relativity/gw150914": dict(
        module="physprior.problems.relativity.gw150914",
        logx=False,
        logy=False,
        param="Mc",
        published="published_Mc_detector",
        extrap="Train on the early inspiral, predict the late",
    ),
    "gravity/kepler": dict(
        module="physprior.problems.gravity.kepler",
        logx=True,
        logy=True,
        param="GM",
        published="published_GM_sun",
        extrap="Train on the terrestrial planets, predict the giants",
    ),
}


def _safe(fn, what):
    try:
        return fn()
    except Exception as e:  # a missing sweep is not fatal
        print(f"    skip {what}: {type(e).__name__}: {str(e)[:90]}")
        return None


def problem_figures(key: str) -> None:
    cfg = TRACKS[key]
    mod = __import__(str(cfg["module"]), fromlist=["problem"])
    prob, meta = mod.problem()
    # problem() carries the problem-level metadata only; the studies run()
    # performed are in results/<track>/meta.json.
    try:
        meta = {**meta, **load_json(key, "meta")}
    except FileNotFoundError:
        print(f"    (no saved meta for {key}; run `physprior run {key.split('/')[0]}`)")
    print(f"  {key}")

    def overview():
        fits = {
            a: fit_arm(a, prob, np.arange(len(prob)), seed=11)
            for a in ("oracle", "physics", "pinn", "sr", "nn")
        }
        fig = P.fig_overview(
            prob,
            fits,
            logx=cfg["logx"],
            logy=cfg["logy"],
            title=f"{key}: every arm, all the data",
        )
        return P.save(fig, key, "overview")

    _safe(overview, "overview")
    for name, (table, xcol, title, xlabel) in {
        "data_efficiency": (
            "sweep_budget",
            "n_train",
            "Held-out error against training budget",
            "training points",
        ),
        "noise": (
            "sweep_noise",
            "noise_frac",
            "Held-out error against added noise (scored on clean values)",
            "added noise / spread of y",
        ),
    }.items():
        _safe(
            lambda t=table, x=xcol, ti=title, xl=xlabel, n=name: P.save(
                P.fig_curve(
                    load_table(key, t), x, "nrmse_out", ti, xl, "held-out nRMSE"
                ),
                key,
                n,
            ),
            name,
        )

    _safe(
        lambda: P.save(
            P.fig_extrapolation_bars(load_table(key, "extrapolation"), cfg["extrap"]),
            key,
            "extrapolation",
        ),
        "extrapolation",
    )
    _safe(
        lambda: P.save(
            P.fig_physics_weight(
                load_table(key, "sweep_physics_weight"),
                "What the physics term in the loss is worth",
                param_key=cfg["param"],
                published=meta[cfg["published"]],
            ),
            key,
            "physics_weight",
        ),
        "physics_weight",
    )

    if key == "relativity/gw150914":
        ab = [r for r in meta.get("pn_ablation", []) if r.get("converged")]
        if ab:
            _safe(
                lambda: P.save(
                    P.fig_bars(
                        [r["label"] for r in ab],
                        [r["Mc"] for r in ab],
                        "Recovered chirp mass against the PN order of the law in the loss",
                        r"$M_c$  [$M_\odot$]",
                        reference=meta["published_Mc_detector"],
                        reference_label="GWTC-1  31.17",
                        fmt="{:.1f}",
                    ),
                    key,
                    "pn_ablation",
                ),
                "pn_ablation",
            )

    if key == "quantum/hydrogen":

        def qed():
            f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
            import matplotlib.pyplot as plt

            P.use_style()
            fig, ax = plt.subplots(figsize=(6.8, 4.0))
            ax.axhline(0, color=P.INK_MUTED, lw=1.2)
            ax.plot(
                prob.x[:, 0],
                prob.y - f.predict(prob.x),
                "o-",
                color=P.ARM_COLOR["physics"],
                lw=1.6,
            )
            P._style(
                ax,
                "Residual of the fitted Bohr law: the structure left is physics",
                "principal quantum number n",
                r"E$_n$ - R(1 - 1/n$^2$)  [cm$^{-1}$]",
            )
            return P.save(fig, key, "bohr_residual")

        _safe(qed, "bohr_residual")


def mercury_figures() -> None:
    """The GR measurement lives in its own problem meta, not in a track."""
    key = "relativity/mercury"
    meta = _safe(lambda: load_json(key, "meta"), "mercury meta")
    if not meta:
        return
    print(f"  {key}")
    lad = meta["gr"]["residual_ladder"]
    _safe(
        lambda: P.save(
            P.fig_bars(
                ["Sun only", "+ all planets", "+ GR (1PN)"],
                [lad["sun_only"], lad["sun_planets"], lad["sun_planets_gr"]],
                "What is left of Mercury's acceleration after each term is removed",
                "mean |residual| / |a|",
                logy=True,
                fmt="{:.1e}",
            ),
            key,
            "gr_residual_ladder",
        ),
        "gr_ladder",
    )
    cv = [r for r in meta.get("gr_convergence", []) if r.get("alpha_GR")]
    if cv:
        _safe(
            lambda: P.save(
                P.fig_bars(
                    [f"{r['step']}/ord{r['fd_order']}" for r in cv],
                    [r["alpha_GR"] for r in cv],
                    "Recovered GR coefficient against derivative accuracy",
                    r"$\alpha$",
                    logy=True,
                    reference=1.0,
                    reference_label="Einstein  1",
                    fmt="{:.4g}",
                ),
                key,
                "gr_convergence",
            ),
            "gr_convergence",
        )


def summary_figures() -> None:
    import json

    h = json.loads((get_settings().results_dir / "headline.json").read_text())
    print("  summary")
    _safe(
        lambda: P.save(
            P.fig_cross_track_extrapolation(
                h["extrapolation_summary"],
                "Error outside the training range, by arm, per track",
            ),
            "summary",
            "cross_track_extrapolation",
        ),
        "cross_track",
    )
    _safe(
        lambda: P.save(
            P.fig_parameter_recovery(h["parameter_recovery"]),
            "summary",
            "parameter_recovery",
        ),
        "param_recovery",
    )
    ctrl = _safe(
        lambda: json.loads(
            (get_settings().results_dir / "_band_control.json").read_text()
        ),
        "band control",
    )
    if ctrl:
        _safe(
            lambda: P.save(
                P.fig_bars(
                    [f"x>={r['x_min']:g}" for r in ctrl],
                    [r["max_rel_dev_outside"] * 100 for r in ctrl],
                    "SR error OUTSIDE the fitted band, against Rayleigh-Jeans coverage",
                    "max deviation from Planck outside band [%]",
                    logy=True,
                    fmt="{:.0f}",
                ),
                "quantum/cmb",
                "band_coverage_control",
            ),
            "band_control",
        )


def inverse_schrodinger_figure(epochs: int = 3000) -> None:
    """Recover V(x) from a spectrum, and draw it.

    Trains, so it is not part of the default figure sweep -- call it directly
    or via `physprior figures --inverse`.
    """
    import numpy as np

    from physprior.methods.eigen_pinn import fit_inverse_potential

    inv = fit_inverse_potential(
        np.arange(6) + 0.5, -6.0, 6.0, n_collocation=256, epochs=epochs, seed=11
    )
    fig = P.fig_inverse_potential(
        inv, truth=lambda x: 0.5 * x**2, title="Recovered: V(x) = x^2/2"
    )
    out = P.save(fig, "quantum", "inverse_potential")
    print(
        f"    wrote {out.relative_to(get_settings().root)}  "
        f"(spectrum error {inv.spectrum_error:.4f})"
    )


def phase2_figure() -> None:
    """What the frozen pinn default bought, per track and question.

    Returns quietly when the before/after table is absent: a fresh clone has
    not re-run the pipeline, and one missing input is not a reason for
    `physprior figures` to stop.
    """
    import pandas as pd

    path = get_settings().results_dir / "phase2_before_after.csv"
    if not path.exists():
        print("    skip phase 2 figure: results/phase2_before_after.csv absent")
        return
    fig = P.fig_phase2_improvement(pd.read_csv(path))
    out = P.save(fig, "", "phase2_improvement")
    print(f"    wrote {out.relative_to(get_settings().root)}")


def main() -> None:
    print("writing figures/")
    for key in TRACKS:
        _safe(lambda k=key: problem_figures(k), f"problem {key}")
    _safe(mercury_figures, "mercury figures")
    summary_figures()
    _safe(phase2_figure, "phase 2 improvement")
    print("done")


if __name__ == "__main__":
    main()
