"""docs/lorenz/README.md, generated from results/lorenz.

Every number on the page comes from a results file through `findings()`,
which the summary notebook, the tutorial and tests/test_lorenz.py also use.
Regenerate with `render_doc()` after `study.run()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json

from . import TRACK
from . import study as ST
from .figures import LABEL
from .system import LAMBDA1, THETA_INIT, THETA_NAMES, THETA_TRUE

DOC = "lorenz/README.md"
MAIN_ARMS = ("nn", "nn_regress", "fd_regress", "shooting", "ms", "pinn", "pinn_polish")


def _fmt(v, nd=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (int, np.integer)):
        return str(v)
    x = float(v)
    if not np.isfinite(x):
        return "–"
    if x == 0:
        return "0"
    return f"{x:.{nd}g}"


def _pct(v, nd=2) -> str:
    return f"{100 * float(v):.{nd}g} %"


def _table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = [r[c] if isinstance(r[c], str) else _fmt(r[c], nd) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def _fig(name: str, alt: str, ext: str = "png") -> str:
    return f"![{alt}](../../figures/{TRACK}/{name}.{ext})"


def main_table() -> pd.DataFrame:
    """Median over the reporting seeds, one row per arm."""
    df = ST.load("main")
    rows = []
    for arm in MAIN_ARMS:
        d = df[df.arm == arm]
        if d.empty:
            continue
        rows.append(
            {
                "method": LABEL[arm],
                "trajectory error": d.state.median() if "state" in d else np.nan,
                "du/dt error": d.deriv.median() if "deriv" in d else np.nan,
                "sigma error": d.err_sigma.median(),
                "rho error": d.err_rho.median(),
                "beta error": d.err_beta.median(),
                "worst seed, constants": d.err_theta.max(),
                "valid forecast (Lyapunov times)": d.vpt.median()
                if "vpt" in d
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ladder_table() -> pd.DataFrame:
    df = ST.load("ladder")
    rows = []
    for name in ST.LADDER:
        d = df[df.label == name]
        rows.append(
            {
                "recipe": name,
                "trajectory error": d.state.median(),
                "constants error": d.err_theta.median(),
                "worst seed, trajectory": d.state.max(),
                "seconds": d.seconds.median(),
            }
        )
    return pd.DataFrame(rows)


def sweep_table(stage: str, x: str) -> pd.DataFrame:
    df = ST.load(stage)
    arms = ("nn", "pinn", "ms", "pinn_polish")
    out = []
    for v, g in df.groupby(x):
        r = {x: v}
        for arm in arms:
            d = g[g.arm == arm]
            r[f"{LABEL[arm]}: trajectory"] = d.state.median()
        for arm in ("nn_regress", "pinn", "ms"):
            d = g[g.arm == arm]
            r[f"{LABEL[arm]}: constants"] = d.err_theta.median()
        out.append(r)
    return pd.DataFrame(out)


def findings() -> dict:
    """The study's numbers that the prose, the notebooks and the tests quote."""
    main = ST.load("main")
    med = main.groupby("arm").median(numeric_only=True)
    worst = main.groupby("arm").max(numeric_only=True)
    lad_all = ST.load("ladder")
    lad = lad_all.groupby("label").median(numeric_only=True)
    noise = ST.load("noise")
    budget = ST.load("budget")
    speed = ST.load("speed")
    meta = load_json(TRACK, "meta")

    def ratio(df, col, a="nn", b="pinn"):
        g = df.groupby("arm")[col].median()
        return float(g[a] / g[b])

    def per_x(df, x, col, a, b):
        out = {}
        for v, g in df.groupby(x):
            ga = g[g.arm == a][col].median()
            gb = g[g.arm == b][col].median()
            out[float(v)] = float(ga / gb)
        return out

    noise_factor = pd.Series(per_x(noise, "noise", "state", "nn", "pinn")).sort_index()
    sp = speed.set_index(
        speed.derivative + np.where(speed.compiled, "+compile", "")
    ).ms_per_step
    ex = ST.load_examples()
    worst_seed = int(
        lad_all[lad_all.label == "vanilla"].set_index("seed").state.idxmax()
    )
    hv = ex[f"ladder/pinn/vanilla/{worst_seed}"]["history"]
    hf = ex[f"main/pinn/chosen/{ST.EXAMPLE_SEED}"]["history"]
    noise_floor = float(meta["reference"]["noise"]) ** 2
    step_f = np.asarray(hf["step"])
    on = step_f >= int(meta["configs"]["pinn"]["warmup"])
    th_f = np.stack([np.asarray(hf[k]) for k in THETA_NAMES], -1) / THETA_TRUE
    close = np.all(np.abs(th_f - 1) < 0.05, axis=1) & on
    f = {
        "vanilla_seed": worst_seed,
        "vanilla_data_over_floor": float(hv["data"][-1]) / noise_floor,
        "vanilla_phys_final": float(hv["phys"][-1]),
        "vanilla_theta_ratio": {
            k: float(hv[k][-1]) / float(v)
            for k, v in zip(THETA_NAMES, THETA_TRUE, strict=True)
        },
        "warmup_data_min_over_floor": float(
            np.min(np.asarray(hf["data"])[~on]) / noise_floor
        ),
        "steps_to_5pct": int(step_f[int(np.argmax(close))] - step_f[on][0])
        if close.any()
        else -1,
        "state_nn": float(med.loc["nn", "state"]),
        "state_pinn": float(med.loc["pinn", "state"]),
        "state_ratio": ratio(main, "state"),
        "deriv_ratio": ratio(main, "deriv"),
        "wins_state": int(
            sum(
                main[(main.arm == "pinn") & (main.seed == s)].state.iloc[0]
                < main[(main.arm == "nn") & (main.seed == s)].state.iloc[0]
                for s in main.seed.unique()
            )
        ),
        "n_seeds": int(main.seed.nunique()),
        "theta_pinn": float(med.loc["pinn", "err_theta"]),
        "theta_pinn_worst": float(worst.loc["pinn", "err_theta"]),
        "theta_nn_regress": float(med.loc["nn_regress", "err_theta"]),
        "theta_fd": float(med.loc["fd_regress", "err_theta"]),
        "theta_shooting": float(med.loc["shooting", "err_theta"]),
        "theta_shooting_best": float(main[main.arm == "shooting"].err_theta.min()),
        "theta_ms": float(med.loc["ms", "err_theta"]),
        "state_ms": float(med.loc["ms", "state"]),
        "state_ms_worst": float(worst.loc["ms", "state"]),
        "state_pinn_worst": float(worst.loc["pinn", "state"]),
        "theta_ms_worst": float(worst.loc["ms", "err_theta"]),
        "theta_polish": float(med.loc["pinn_polish", "err_theta"]),
        "vpt_pinn": float(med.loc["pinn", "vpt"]),
        "vpt_polish": float(med.loc["pinn_polish", "vpt"]),
        "vpt_ms": float(med.loc["ms", "vpt"]),
        "vpt_nn": float(med.loc["nn", "vpt"]),
        "vpt_nn_regress": float(med.loc["nn_regress", "vpt"]),
        "ladder_vanilla": float(lad.loc["vanilla", "state"]),
        "ladder_final": float(lad.loc[ST.LADDER_FINAL, "state"]),
        "ladder_gain": float(
            lad.loc["vanilla", "state"] / lad.loc[ST.LADDER_FINAL, "state"]
        ),
        "ladder_lbfgs_gain": float(
            lad.loc["+ cosine lr, resampling", "state"]
            / lad.loc["+ L-BFGS finish", "state"]
        ),
        "ladder_gradnorm": float(lad.loc["vanilla + gradnorm", "state"]),
        "ladder_causal": float(lad.loc["final + causal", "state"]),
        "noise_ratio": noise_factor.to_dict(),
        "budget_ratio": per_x(budget, "n_obs", "state", "nn", "pinn"),
        "noise_theta_pinn": noise[noise.arm == "pinn"]
        .groupby("noise")
        .err_theta.median()
        .to_dict(),
        "noise_theta_nn_regress": noise[noise.arm == "nn_regress"]
        .groupby("noise")
        .err_theta.median()
        .to_dict(),
        "budget_theta_pinn": budget[budget.arm == "pinn"]
        .groupby("n_obs")
        .err_theta.median()
        .to_dict(),
        "budget_theta_ms": budget[budget.arm == "ms"]
        .groupby("n_obs")
        .err_theta.median()
        .to_dict(),
        "budget_theta_nn_regress": budget[budget.arm == "nn_regress"]
        .groupby("n_obs")
        .err_theta.median()
        .to_dict(),
        "speed_forward": float(sp["forward"]),
        "speed_autograd": float(sp["autograd"]),
        "speed_jvp": float(sp["jvp"]),
        "speed_compile": float(sp.get("forward+compile", np.nan)),
        "speedup_forward": float(sp["autograd"] / sp["forward"]),
        "lambda_fit": float(meta["chaos"]["lambda_fit"]),
        "lambda_ref": float(LAMBDA1),
        "choice": meta["choice"],
        "n_obs": int(meta["reference"]["n_obs"]),
        "noise": float(meta["reference"]["noise"]),
        "t_end": float(meta["reference"]["t_end"]),
        "shooting_fails": int((main[main.arm == "shooting"].err_theta > 0.1).sum()),
        "vanilla_fails": int((lad_all[lad_all.label == "vanilla"].state > 0.1).sum()),
        "ladder_worst": lad_all.groupby("label").state.max().to_dict(),
        "noise_factor_clean": float(noise_factor.iloc[0]),
        "noise_factor_max": float(noise_factor.iloc[-1]),
        "noise_max": float(noise_factor.index[-1]),
        "noise_state_nn": noise[noise.arm == "nn"]
        .groupby("noise")
        .state.median()
        .to_dict(),
        "noise_state_pinn": noise[noise.arm == "pinn"]
        .groupby("noise")
        .state.median()
        .to_dict(),
        "compile_gain": float(sp["forward"] / sp["forward+compile"])
        if "forward+compile" in sp
        else float("nan"),
        "noise_monotone": bool(
            np.all(np.diff(noise_factor[noise_factor.index > 0]) > 0)
        ),
    }
    return f


def render_doc(path=None) -> str:
    f = findings()
    meta = load_json(TRACK, "meta")
    ref = meta["reference"]
    th0 = ", ".join(
        f"{n} = {v:g}" for n, v in zip(THETA_NAMES, THETA_INIT, strict=True)
    )
    tht = ", ".join(
        f"{n} = {v:.4g}" for n, v in zip(THETA_NAMES, THETA_TRUE, strict=True)
    )
    parts = [
        "# Lorenz-63: a PINN against a black box, on a chaotic inverse problem",
        "",
        "*Generated by `physprior lorenz` from `results/lorenz/`; do not edit by hand.*",
        "",
        _fig("butterfly_3d", "two trajectories 1e-8 apart"),
        "",
        "## The problem",
        "",
        f"One Lorenz-63 trajectory over t in [0, {ref['t_end']:g}] (about "
        f"{ref['t_end'] * LAMBDA1:.1f} Lyapunov times), observed at "
        f"{ref['n_obs']} random times with Gaussian noise of "
        f"{_pct(ref['noise'])} of each component's spread. From these "
        "observations alone:",
        "",
        "1. reconstruct x(t), y(t), z(t) between the observations;",
        f"2. recover the three constants ({tht}), starting from {th0};",
        "3. forecast past the end of the window.",
        "",
        "Every hyperparameter was chosen on the tuning seeds 3 / 7 / 19; every "
        "number below is on the reporting seeds 11 / 23 / 42 (median, with the "
        "worst seed where it matters).",
        "",
        "## The method",
        "",
        _fig("block_diagram", "PINN block diagram"),
        "",
        "```mermaid",
        "flowchart LR",
        '  t[time t] --> F[Fourier features] --> N[MLP, tanh] --> U["u(t) = mu + sd N"]',
        '  U --> LD["L_data: fit to 40 noisy observations"]',
        '  N -- forward-mode dN/ds --> D["du/dt"]',
        '  U --> R["residual du/dt - f(u; sigma, rho, beta)"]',
        "  D --> R",
        '  TH["sigma, rho, beta (trained)"] --> R',
        "  R --> LP[L_phys]",
        '  LD --> L["L = L_data + w(k) L_phys"]',
        "  LP --> L",
        '  L --> O["Adam, cosine lr -> L-BFGS"]',
        "```",
        "",
        "The network maps time to state. The data term fits the observations; the "
        "physics term is the residual of the Lorenz equations at 256 collocation "
        "times redrawn every step, with the three constants as trainable "
        "parameters. The time derivative is carried through the network in "
        "forward mode, in the same pass as the state.",
        "",
        "The black box is the same kind of network trained on the data term "
        f"alone, its size and features tuned on the tuning seeds (chosen: "
        f"`{f['choice']['nn']}`; PINN: `{f['choice']['pinn']}`; multiple "
        f"shooting: `{f['choice']['ms']}`).",
        "",
        "## Results",
        "",
        _fig("main", "every method on the reference problem"),
        "",
        _table(main_table()),
        "",
        f"- **Trajectory.** The PINN's reconstruction error is "
        f"{_fmt(f['state_pinn'])} against the black box's {_fmt(f['state_nn'])}, "
        f"a factor {f['state_ratio']:.1f} (better on {f['wins_state']} of "
        f"{f['n_seeds']} seeds). For du/dt the factor is {f['deriv_ratio']:.1f}.",
        f"- **Constants.** From a start that is off by a factor of about two in "
        f"each constant, the PINN recovers them to {_pct(f['theta_pinn'])} "
        f"(worst seed {_pct(f['theta_pinn_worst'])}). Single shooting from the "
        f"same start lands in a wrong local minimum on "
        f"{f['shooting_fails']} of {f['n_seeds']} seeds (median error "
        f"{_pct(f['theta_shooting'], 3)}). The black box, differentiated and "
        f"regressed on the law, reaches {_pct(f['theta_nn_regress'])}; finite "
        f"differences of the raw data reach {_pct(f['theta_fd'])}.",
        f"- **Against the classical remedy.** Multiple shooting, the standard "
        f"method for chaotic systems, recovers the constants to "
        f"{_pct(f['theta_ms'])} (worst seed {_pct(f['theta_ms_worst'])}) once its "
        "segment count is tuned; with too few segments it fails like single "
        "shooting (see the tuning table). On the constants it is as good as the "
        "PINN. On the trajectory it is as good in the median "
        f"({_fmt(f['state_ms'])}) but not on every seed: its worst is "
        f"{_fmt(f['state_ms_worst'])} against the PINN's "
        f"{_fmt(f['state_pinn_worst'])}, because one trajectory integrated from "
        "the fitted start compounds its errors chaotically. Single shooting "
        f"started from the PINN's answer reaches {_pct(f['theta_polish'])}. With "
        "few observations the PINN is clearly ahead (next section).",
        f"- **Forecast.** Integrating the law with the PINN's constants from its "
        f"own end state gives {_fmt(f['vpt_pinn'], 2)} Lyapunov times of valid "
        f"forecast, {_fmt(f['vpt_polish'], 2)} after polishing; multiple shooting "
        f"gives {_fmt(f['vpt_ms'], 2)}. The black box extrapolated gives "
        f"{_fmt(f['vpt_nn'], 2)}. The horizon is set by the error in the end "
        "state and the constants, amplified at the Lyapunov rate.",
        "",
        _fig("reconstruction", "reconstruction on seed 11"),
        "",
        _fig("phase_space", "reconstruction in phase space"),
        "",
        "## Noise and data budget",
        "",
        _fig("noise", "noise sweep"),
        "",
        _table(sweep_table("noise", "noise")),
        "",
        _fig("budget", "budget sweep"),
        "",
        _table(sweep_table("budget", "n_obs")),
        "",
        f"**Noise.** The black box cannot interpolate a chaotic trajectory "
        f"between {f['n_obs']} points even without noise: its error is "
        f"{_fmt(f['noise_state_nn'][0.0])} at zero noise. The PINN's error "
        "scales with the noise, since the law fills in between the points. "
        f"The factor between them is {f['noise_factor_clean']:.0f} without "
        f"noise and {f['noise_factor_max']:.1f} at {_pct(f['noise_max'])} noise.",
        "",
        f"**Data budget.** The factor is "
        f"{f['budget_ratio'][max(f['budget_ratio'])]:.1f} at "
        f"{max(f['budget_ratio']):g} observations and largest, "
        f"{max(f['budget_ratio'].values()):.0f}, at "
        f"{max(f['budget_ratio'], key=f['budget_ratio'].get):g}. At "
        f"{min(f['budget_ratio']):g} observations neither network reconstructs "
        "the trajectory, but the PINN still recovers the constants to "
        f"{_pct(f['budget_theta_pinn'][min(f['budget_ratio'])])}, against "
        f"{_pct(f['budget_theta_ms'][min(f['budget_ratio'])])} for multiple "
        "shooting and "
        f"{_pct(f['budget_theta_nn_regress'][min(f['budget_ratio'])])} for the "
        "black box followed by regression.",
        "",
        "## Optimising the PINN",
        "",
        _fig("ladder", "the optimisation ladder"),
        "",
        _table(ladder_table()),
        "",
        f"The vanilla PINN (constant learning rate, physics weight on from step "
        f"0, fixed collocation grid, Adam only) fails on {f['vanilla_fails']} of "
        f"{f['n_seeds']} seeds (worst trajectory error "
        f"{_fmt(f['ladder_worst']['vanilla'])}). On its worst seed it stalls in a "
        "poor minimum that satisfies neither term: the data loss ends "
        f"{f['vanilla_data_over_floor']:.0f} times the noise floor, the "
        f"residual at {_fmt(f['vanilla_phys_final'], 2)}, and sigma, rho, beta at "
        + ", ".join(f"{v:.2f}" for v in f["vanilla_theta_ratio"].values())
        + " of their true values. The data "
        "warm-up with a ramped physics weight is the piece that matters most. "
        "The later rungs change the median little and tighten the worst seed: "
        f"the final recipe's worst seed is "
        f"{_fmt(f['ladder_worst'][ST.LADDER_FINAL])}, against "
        f"{_fmt(f['ladder_worst']['+ warm-up and ramp'])} for the warm-up alone. "
        "The L-BFGS finish improves "
        f"the rung before it by a factor {f['ladder_lbfgs_gain']:.1f}. "
        "Gradient-norm balancing, added to the vanilla recipe instead of the "
        f"warm-up, brings it to {_fmt(f['ladder_gradnorm'])} (worst seed "
        f"{_fmt(f['ladder_worst']['vanilla + gradnorm'])}); causal weights on "
        f"top of the final recipe give {_fmt(f['ladder_causal'])}, no change.",
        "",
        "### Where the loss goes",
        "",
        _fig("loss_breakdown", "loss terms during training"),
        "",
        "Top left: the data term and the physics term of each equation. During "
        "the warm-up the data term falls to "
        f"{f['warmup_data_min_over_floor']:.2f} of the noise floor: alone, the "
        "network fits the noise. Within "
        f"{f['steps_to_5pct']} steps of the physics term coming on, all three "
        "constants are within 5 % of the truth (bottom left), and the data term "
        "returns to the noise floor and stays there. For comparison, the "
        f"vanilla recipe on seed {f['vanilla_seed']}:",
        "",
        _fig("loss_breakdown_vanilla", "loss terms of the vanilla PINN"),
        "",
        "### What the derivative costs",
        "",
        _fig("speed", "ms per step by derivative method"),
        "",
        _table(
            ST.load("speed")[
                [
                    "derivative",
                    "compiled",
                    "ms_per_step",
                    "max_abs_diff_vs_autograd",
                    "torch",
                ]
            ]
        ),
        "",
        f"The time derivative computed by hand in forward mode costs "
        f"{_fmt(f['speed_forward'], 2)} ms per step against "
        f"{_fmt(f['speed_autograd'], 2)} ms for reverse-mode autograd "
        f"({f['speedup_forward']:.1f}x) and {_fmt(f['speed_jvp'], 2)} ms for "
        "`torch.func.jvp`; all three agree to rounding. `torch.compile` on the "
        f"forward-mode step gives a further factor {f['compile_gain']:.2f} "
        f"({_fmt(f['speed_compile'], 2)} ms). On a CPU at this size the step is "
        "dominated by per-operation overhead, so fusing operations helps and "
        "fewer operations help more.",
        "",
        "## The butterfly effect",
        "",
        _fig("butterfly", "an ensemble spreading over the attractor", "gif"),
        "",
        _fig("separation", "separation growth"),
        "",
        f"Neighbouring trajectories separate like exp(lambda_1 t). The slope of "
        f"the mean log-separation is {f['lambda_fit']:.3f}, against the reference "
        f"lambda_1 = {f['lambda_ref']}. That is why a forecast has a horizon "
        "whatever the method: an error e in the state grows to the validity "
        "threshold 0.4 in about ln(0.4 / e) / lambda_1.",
        "",
        _fig("horizon", "forecast horizon vs error in the constants"),
        "",
        _fig("forecast", "forecast error after the window"),
        "",
        "## Reproduce",
        "",
        "```bash",
        "physprior lorenz            # tune, run, figures, this page (about an hour on 8 cores)",
        "physprior lorenz --quick    # the same pipeline with short training",
        "physprior lorenz --doc-only # figures and this page from results/lorenz",
        "```",
        "",
        "Tuning grid and every run: `results/lorenz/*.csv`; configuration and "
        "choices: `results/lorenz/meta.json`. RK4 step study for the shooting "
        "and forecast integrator: `results/lorenz/rk4_convergence.csv`.",
        "",
        "## References",
        "",
        "- E. N. Lorenz, Deterministic nonperiodic flow, J. Atmos. Sci. 20, 130 (1963).",
        "- M. Raissi, P. Perdikaris, G. E. Karniadakis, Physics-informed neural "
        "networks, J. Comput. Phys. 378, 686 (2019).",
        "- H. G. Bock, Numerical treatment of inverse problems in chemical "
        "reaction kinetics, Springer Ser. Chem. Phys. 18, 102 (1981).",
        "- S. Wang, Y. Teng, P. Perdikaris, Understanding and mitigating gradient "
        "flow pathologies in physics-informed neural networks, SIAM J. Sci. "
        "Comput. 43, A3055 (2021).",
        "- S. Wang, S. Sankaran, P. Perdikaris, Respecting causality for training "
        "physics-informed neural networks, Comput. Methods Appl. Mech. Eng. 421, "
        "116813 (2024).",
        "- M. Tancik et al., Fourier features let networks learn high frequency "
        "functions in low dimensional domains, NeurIPS (2020).",
        "- J. Pathak et al., Model-free prediction of large spatiotemporally "
        "chaotic systems from data, Phys. Rev. Lett. 120, 024102 (2018).",
        "",
    ]
    text = "\n".join(parts)
    out = path or get_settings().root / "docs" / DOC
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text
