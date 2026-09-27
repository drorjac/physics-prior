"""Run the optimization study, draw its figures, and write its README.

    from physprior.optim import report
    report.run_all(quick=False)     # every study, then figures and the doc
    report.make_figures()           # from results/optim only
    report.render_doc()             # docs/optimization/README.md

Every number in the generated README is read from results/optim; nothing is
typed by hand.
"""

from __future__ import annotations

import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json, load_table, save_table
from physprior.viz import plots as P

AREA = "optim"

# Optimizer colours. The four first-order methods take the repo's four
# validated categorical slots (all pairs, light surface: see viz/plots.py);
# the two second-order methods are drawn in ink with their own line styles
# and markers, so no fifth or sixth hue is needed -- six hues from this
# palette do not pass the all-pairs CVD gate.
OPT_STYLE = {
    "sgd": {"color": "#2a78d6", "ls": "-", "marker": "o"},
    "momentum": {"color": "#eb6834", "ls": "-", "marker": "s"},
    "rmsprop": {"color": "#1baf7a", "ls": "-", "marker": "^"},
    "adam": {"color": "#4a3aa7", "ls": "-", "marker": "D"},
    "lbfgs": {"color": P.INK, "ls": "--", "marker": "v"},
    "lm": {"color": P.INK_2, "ls": ":", "marker": "x"},
}
OPT_ORDER = ["sgd", "momentum", "rmsprop", "adam", "lbfgs", "lm"]
MODEL_COLOR = {k: P.ARM_COLOR[k] for k in ("physics", "pinn", "nn")}
MODEL_LABEL = {
    "physics": "physics (law, constants fitted)",
    "pinn": "PINN",
    "nn": "black-box NN",
}
TASK_LABEL = {
    "hydrogen": "hydrogen (NIST, Bohr law)",
    "oscillator": "damped oscillator (ODE)",
    "heat": "heat equation (PDE)",
}


# ---------------------------------------------------------------------------
# the from-scratch network
# ---------------------------------------------------------------------------


def init_depth_scan(depth: int = 8, width: int = 64, n: int = 512) -> pd.DataFrame:
    """Std of the activations layer by layer, for each init scheme, on a tanh
    and a ReLU network: why the init is matched to the activation."""
    from physprior.optim import scratch as S

    rows = []
    x = np.random.default_rng(0).normal(size=(n, width))
    for act in ("tanh", "relu"):
        for scheme in ("xavier", "he", "naive"):
            net = S.init_mlp([width] * (depth + 1) + [1], act, scheme, seed=0)
            _, cache = S.forward(net, x)
            for layer, a in enumerate(cache["a"][1:-1], start=1):
                rows.append(
                    {
                        "activation": act,
                        "init": scheme,
                        "layer": layer,
                        "std": float(a.std()),
                        # tanh units in the flat tails pass almost no gradient
                        "saturated_frac": float(np.mean(np.abs(a) > 0.99))
                        if act == "tanh"
                        else float(np.mean(a == 0)),
                    }
                )
    return pd.DataFrame(rows)


def run_scratch(quick: bool = False) -> dict[str, pd.DataFrame]:
    from physprior.optim import scratch as S

    net = S.init_mlp([1, 16, 16, 1], "tanh", seed=0)
    x, y, _ = S.demo_data(32)
    _, g = S.loss_and_grad(net, x, y)
    fd = pd.DataFrame(S.fd_convergence(net, x, y))
    fd["torch_rel_error"] = float(
        np.linalg.norm(S.torch_grad(net, x, y) - g) / np.linalg.norm(g)
    )
    save_table(fd, AREA, "scratch_gradcheck")
    res = S.compare_optimizers(steps=300 if quick else 3000)
    curves = pd.concat(
        [
            pd.DataFrame(v["history"]).assign(optimizer=k, lr=v["lr"])
            for k, v in res.items()
        ],
        ignore_index=True,
    )
    save_table(curves, AREA, "scratch_optimizers")
    scan = init_depth_scan()
    save_table(scan, AREA, "scratch_init")
    return {"gradcheck": fd, "curves": curves, "init": scan}


# ---------------------------------------------------------------------------
# everything
# ---------------------------------------------------------------------------


def run_all(quick: bool = False, workers: int | None = None) -> None:
    import torch

    from physprior.optim import losses_study, optimizers_study, pinn_balance

    torch.set_num_threads(2)
    steps = [
        ("scratch network", lambda: run_scratch(quick)),
        ("optimizers", lambda: optimizers_study.run(quick, workers=workers)),
        ("loss functions", lambda: losses_study.run(quick, workers=workers)),
        ("pinn balance", lambda: pinn_balance.run(quick, workers=workers)),
    ]
    for name, fn in steps:
        t0 = time.time()
        print(f"[optim] {name}", flush=True)
        fn()
        print(f"[optim] {name} done in {time.time() - t0:.0f}s", flush=True)
    make_figures()
    render_doc()


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def _save(fig, name: str):
    return P.save(fig, AREA, name)


def fig_scratch() -> None:
    P.use_style()
    c = load_table(AREA, "scratch_optimizers")
    fd = load_table(AREA, "scratch_gradcheck")
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for opt in ("sgd", "momentum", "rmsprop", "adam"):
        d = c[c.optimizer == opt]
        st = OPT_STYLE[opt]
        ax[0].plot(
            d.step,
            d.loss,
            color=st["color"],
            lw=1.6,
            label=f"{opt} (lr {d.lr.iloc[0]:g})",
        )
    d = c[c.optimizer == "nesterov"]
    ax[0].plot(
        d.step,
        d.loss,
        color=P.INK,
        ls="--",
        lw=1.4,
        label=f"nesterov (lr {d.lr.iloc[0]:g})",
    )
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=8)
    P._style(ax[0], "hand-written optimizers, same initial network", "step", "MSE")
    ax[1].loglog(
        fd.h,
        fd.rel_error,
        marker="o",
        color=P.ARM_COLOR["physics"],
        label="backprop vs central difference",
    )
    ax[1].axhline(
        fd.torch_rel_error.iloc[0] + 1e-18,
        color=P.INK_MUTED,
        ls="--",
        lw=1.2,
        label="backprop vs torch autograd",
    )
    hs = fd.h.to_numpy()
    ref = fd.rel_error.iloc[0] * (hs / hs[0]) ** 2
    ax[1].loglog(hs, ref, color=P.INK_2, ls=":", lw=1.2, label="slope 2 (O(h$^2$))")
    ax[1].legend(fontsize=8)
    P._style(ax[1], "gradient check", "finite-difference step h", "relative error")
    _save(fig, "01_scratch_optimizers_gradcheck")


def fig_init() -> None:
    P.use_style()
    s = load_table(AREA, "scratch_init")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    styles = {
        "xavier": ("#2a78d6", "o"),
        "he": ("#eb6834", "s"),
        "naive": ("#4a3aa7", "^"),
    }
    for ax, act in zip(axes, ("tanh", "relu"), strict=True):
        for scheme, (col, mk) in styles.items():
            d = s[(s.activation == act) & (s.init == scheme)]
            ax.plot(d.layer, d["std"], color=col, marker=mk, label=scheme)
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        P._style(ax, f"{act} network, width 64", "layer", "std of activations")
    _save(fig, "02_init_scan")


def fig_lr_sensitivity() -> None:
    P.use_style()
    g = load_table(AREA, "lr_grid")
    tasks = [t for t in TASK_LABEL if t in set(g.task)]
    models = [m for m in MODEL_LABEL if m in set(g.model)]
    fig, axes = plt.subplots(
        len(models),
        len(tasks),
        figsize=(4 * len(tasks), 3.1 * len(models)),
        squeeze=False,
    )
    g = g.copy()
    g["score"] = g.final_data_loss.where(~g.diverged, np.nan)
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i, j]
            d = g[(g.task == t) & (g.model == m)]
            for opt in OPT_ORDER:
                e = d[d.optimizer == opt].groupby("lr").score.median()
                if len(e):
                    st = OPT_STYLE[opt]
                    ax.plot(
                        e.index,
                        e.values,
                        color=st["color"],
                        ls=st["ls"],
                        marker=st["marker"],
                        ms=5,
                        lw=1.5,
                        label=opt,
                    )
            tol = d.tol.iloc[0] if len(d) else np.nan
            ax.axhline(tol, color=P.INK_MUTED, ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_yscale("log")
            P._style(
                ax,
                f"{TASK_LABEL[t]}\n{MODEL_LABEL[m]}",
                "learning rate",
                "final data loss",
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        "Learning-rate sensitivity, tuning seeds (median; gaps = diverged; dashed = tolerance)",
        fontsize=10,
    )
    _save(fig, "03_lr_sensitivity")


def fig_loss_curves() -> None:
    P.use_style()
    c = load_table(AREA, "curves")
    tasks = [t for t in TASK_LABEL if t in set(c.task)]
    models = [m for m in MODEL_LABEL if m in set(c.model)]
    seed = int(c.seed.min())
    fig, axes = plt.subplots(
        len(models),
        len(tasks),
        figsize=(4 * len(tasks), 3.1 * len(models)),
        squeeze=False,
    )
    for i, m in enumerate(models):
        for j, t in enumerate(tasks):
            ax = axes[i, j]
            d = c[(c.task == t) & (c.model == m) & (c.seed == seed)]
            for opt in OPT_ORDER:
                e = d[d.optimizer == opt]
                if len(e):
                    st = OPT_STYLE[opt]
                    ax.plot(
                        e.evals,
                        e.data_loss,
                        color=st["color"],
                        ls=st["ls"],
                        lw=1.4,
                        label=opt,
                    )
            ax.set_yscale("log")
            P._style(
                ax,
                f"{TASK_LABEL[t]}\n{MODEL_LABEL[m]}",
                "loss + gradient evaluations",
                "data loss",
            )
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle(
        f"Training curves at the rate chosen on the tuning seeds (reporting seed {seed})",
        fontsize=10,
    )
    _save(fig, "04_loss_curves")


def fig_spectra() -> None:
    P.use_style()
    eig = load_json(AREA, "hessian_eigenvalues")
    tasks = [t for t in TASK_LABEL if t in {e["task"] for e in eig}]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(4 * len(tasks), 3.6), squeeze=False
    )
    for j, t in enumerate(tasks):
        ax = axes[0, j]
        for m in MODEL_LABEL:
            rows = [
                e
                for e in eig
                if e["task"] == t and e["model"] == m and e["optimizer"] == "adam"
            ]
            for k, e in enumerate(rows):
                v = np.sort(np.abs(np.asarray(e["eigenvalues"])))[::-1]
                ax.loglog(
                    np.arange(1, len(v) + 1),
                    v,
                    color=MODEL_COLOR[m],
                    lw=1.3,
                    alpha=0.9,
                    marker="o" if len(v) < 5 else None,
                    label=MODEL_LABEL[m] if k == 0 else None,
                )
        ax.legend(fontsize=7)
        P._style(ax, TASK_LABEL[t], "eigenvalue rank", "|Hessian eigenvalue|")
    fig.suptitle(
        "Hessian spectrum of the training loss at the Adam solution (reporting seeds)",
        fontsize=10,
    )
    _save(fig, "05_hessian_spectra")


def fig_grad_ratio() -> None:
    P.use_style()
    c = load_table(AREA, "curves")
    d = c[(c.model == "pinn") & (c.optimizer == "adam")].dropna(
        subset=["grad_data", "grad_phys"]
    )
    tasks = [t for t in TASK_LABEL if t in set(d.task)]
    fig, axes = plt.subplots(
        1, len(tasks), figsize=(4 * len(tasks), 3.4), squeeze=False
    )
    for j, t in enumerate(tasks):
        ax = axes[0, j]
        for k, (seed, e) in enumerate(d[d.task == t].groupby("seed")):
            ratio = e.grad_phys / e.grad_data
            ax.plot(
                e.evals,
                ratio,
                color=MODEL_COLOR["pinn"],
                lw=1.3,
                alpha=[1, 0.7, 0.45][k % 3],
                label=f"seed {seed}",
            )
        ax.axhline(1.0, color=P.INK_MUTED, ls="--", lw=1)
        ax.set_yscale("log")
        ax.legend(fontsize=7)
        P._style(
            ax,
            TASK_LABEL[t],
            "evaluations (Adam)",
            "||grad w L_phys|| / ||grad L_data||",
        )
    fig.suptitle(
        "Gradient-norm ratio of the two loss terms over the network weights",
        fontsize=10,
    )
    _save(fig, "06_grad_ratio")


def fig_losses() -> None:
    P.use_style()
    s = load_table(AREA, "losses_summary")
    losses = ["mse", "l1", "huber", "cauchy", "gauss_nll"]
    noises = ["gaussian", "student_t", "outliers"]
    cols = {"gaussian": "#2a78d6", "student_t": "#eb6834", "outliers": "#1baf7a"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    xs = np.arange(len(losses))
    for k, nz in enumerate(noises):
        off = (k - 1) * 0.25
        p = (
            s[(s.model == "physics") & (s.noise == nz)]
            .set_index("loss")
            .reindex(losses)
        )
        axes[0].errorbar(
            xs + off,
            p.gamma_bias_pct,
            yerr=p.gamma_sd_pct,
            fmt="o",
            color=cols[nz],
            capsize=3,
            label=nz,
        )
        for ax, m in ((axes[1], "physics"), (axes[2], "nn")):
            q = s[(s.model == m) & (s.noise == nz)].set_index("loss").reindex(losses)
            ax.plot(
                xs + off,
                q.nrmse_in,
                "o",
                color=cols[nz],
                label=f"{nz}, in range" if m == "physics" else None,
            )
            ax.plot(
                xs + off,
                q.nrmse_out,
                "^",
                color=cols[nz],
                mfc="none",
                label=f"{nz}, out of range" if m == "physics" else None,
            )
    axes[0].axhline(0, color=P.INK_MUTED, lw=1)
    for ax in axes:
        ax.set_xticks(xs, losses)
    for ax in axes[1:]:
        ax.set_yscale("log")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    P._style(axes[0], "physics: error in gamma (mean +- sd)", None, "gamma error [%]")
    P._style(axes[1], "physics: prediction error", None, "nrmse vs clean truth")
    P._style(axes[2], "black-box NN: prediction error", None, "nrmse vs clean truth")
    _save(fig, "07_loss_functions")


def fig_w_dial() -> None:
    P.use_style()
    w = load_table(AREA, "w_phys_dial")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    cols = {"gaussian": "#2a78d6", "outliers": "#1baf7a"}
    for nz, d in w.groupby("noise"):
        m = d.groupby("w_phys").median(numeric_only=True)
        x = np.where(m.index > 0, m.index, 1e-4)
        for ax, col in zip(
            axes, ("nrmse_in", "nrmse_out", "err_gamma_pct"), strict=True
        ):
            y = np.abs(m[col]) if col.startswith("err") else m[col]
            ax.plot(x, y, marker="o", color=cols[nz], label=nz)
    for ax, (t, yl) in zip(
        axes,
        (
            ("in range", "nrmse"),
            ("out of range", "nrmse"),
            ("recovered damping", "|gamma error| [%]"),
        ),
        strict=True,
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        P._style(ax, t, "w_phys (w = 0 drawn at 1e-4)", yl)
    fig.suptitle(
        "The physics weight as a loss hyperparameter: residual PINN, damped oscillator (reporting seeds, median)",
        fontsize=10,
    )
    _save(fig, "08_w_phys_dial")


def fig_balance() -> None:
    P.use_style()
    h = load_table(AREA, "balance_history")
    tracks = list(dict.fromkeys(h.track))
    fig, axes = plt.subplots(
        2, len(tracks), figsize=(3.6 * len(tracks), 6.2), squeeze=False
    )
    for j, tr in enumerate(tracks):
        for split, ls in (("interp", "-"), ("extrap", "--")):
            for var, col in (
                ("balanced", MODEL_COLOR["pinn"]),
                ("unbalanced", MODEL_COLOR["nn"]),
            ):
                d = h[(h.track == tr) & (h.split == split) & (h.variant == var)]
                for k, (_, e) in enumerate(d.groupby("seed")):
                    lab = f"{var}, {split}" if k == 0 else None
                    if var == "balanced":
                        axes[0, j].plot(
                            e.epoch,
                            e.w_phys,
                            color=col,
                            ls=ls,
                            lw=1.2,
                            alpha=0.8,
                            label=lab,
                        )
                    axes[1, j].plot(
                        e.epoch,
                        e.correction_rms_frac,
                        color=col,
                        ls=ls,
                        lw=1.2,
                        alpha=0.8,
                        label=lab,
                    )
        axes[0, j].set_yscale("log")
        axes[1, j].set_yscale("log")
        P._style(axes[0, j], tr, "epoch", "annealed w_phys")
        P._style(axes[1, j], None, "epoch", "correction RMS / sd_y (train)")
        if j == 0:
            axes[0, j].legend(fontsize=7)
            axes[1, j].legend(fontsize=7)
    fig.suptitle(
        "`balance`: the physics weight and the correction during training (tuning seeds 3/7/19)",
        fontsize=10,
    )
    _save(fig, "09_balance_history")

    s = load_table(AREA, "balance_summary")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    labels = [
        f"{t}\n{sp}" for t, sp in dict.fromkeys(zip(s.track, s.split, strict=True))
    ]
    keys = list(dict.fromkeys(zip(s.track, s.split, strict=True)))
    xs = np.arange(len(keys))
    for k, (var, col) in enumerate(
        (
            ("physics", MODEL_COLOR["physics"]),
            ("balanced", MODEL_COLOR["pinn"]),
            ("unbalanced", MODEL_COLOR["nn"]),
        )
    ):
        for ax, c in zip(axes, ("nrmse_in", "nrmse_out"), strict=True):
            v = [
                s[(s.track == t) & (s.split == sp) & (s.variant == var)][c].median()
                for t, sp in keys
            ]
            ax.bar(xs + (k - 1) * 0.27, v, width=0.25, color=col, label=var)
    for ax, t in zip(axes, ("training points", "held-out points"), strict=True):
        ax.set_yscale("log")
        ax.set_xticks(xs, labels, fontsize=7)
        ax.legend(fontsize=8)
        P._style(ax, t, None, "nrmse (median, tuning seeds)")
    _save(fig, "10_balance_errors")


def make_figures() -> None:
    for fn in (
        fig_scratch,
        fig_init,
        fig_lr_sensitivity,
        fig_loss_curves,
        fig_spectra,
        fig_grad_ratio,
        fig_losses,
        fig_w_dial,
        fig_balance,
    ):
        try:
            fn()
        except FileNotFoundError as e:
            print(f"  skipped {fn.__name__}: {e}")


# ---------------------------------------------------------------------------
# the README, generated
# ---------------------------------------------------------------------------


def _fmt(v, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—" if v is None or np.isnan(v) else "inf"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    if v == 0:
        return "0"
    if abs(v) >= 1e4 or abs(v) < 1e-3:
        return f"{v:.{digits - 1}e}"
    return f"{v:.{digits}g}"


def _table(df: pd.DataFrame, cols: list[str], headers: list[str] | None = None) -> str:
    headers = headers or cols
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(cols)]
    for r in df[cols].itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(_fmt(v) if not isinstance(v, str) else v for v in r)
            + " |"
        )
    return "\n".join(lines)


def _fig(name: str) -> str:
    return f"![{name}](../../figures/optim/{name}.png)"


def optimizer_findings(report: pd.DataFrame) -> dict:
    """The comparisons the README states, computed rather than typed."""
    from physprior.optim.optimizers_study import summarise

    s = summarise(report)
    out: dict = {}
    for t in s.task.unique():
        d = s[s.task == t]
        best = {}
        for m in d.model.unique():
            e = d[(d.model == m) & (d.failure_rate < 1)]
            if len(e):
                r = e.sort_values(["evals_to_tol", "final_data_loss"]).iloc[0]
                best[m] = (r.optimizer, r.evals_to_tol)
        out[t] = best
    return out


def render_doc(path=None) -> str:
    """Write docs/optimization/README.md (or `path`) from results/optim."""
    from physprior.optim.optimizers_study import summarise

    settings = get_settings()
    parts: list[str] = [_HEADER]

    # -- scratch
    try:
        fd = load_table(AREA, "scratch_gradcheck")
        parts.append(_SCRATCH_TEXT)
        parts.append(
            f"Backprop against torch autograd: relative difference "
            f"{_fmt(fd.torch_rel_error.iloc[0])}. Against central differences the "
            f"error falls from {_fmt(fd.rel_error.iloc[0])} at h = {_fmt(fd.h.iloc[0])} "
            f"to a minimum of {_fmt(fd.rel_error.min())} at h = "
            f"{_fmt(fd.h.iloc[int(fd.rel_error.idxmin())])}, then rises as round-off "
            f"takes over: the O(h^2) truncation line, then the eps/h floor.\n"
        )
        c = load_table(AREA, "scratch_optimizers")
        fin = c.groupby("optimizer").loss.last()
        parts.append(
            "Final training MSE after the same number of steps (one default "
            "rate each, not tuned): "
            + ", ".join(
                f"{k} {_fmt(fin[k])}"
                for k in ("sgd", "momentum", "nesterov", "rmsprop", "adam")
                if k in fin
            )
            + ".\n"
        )
        sc = load_table(AREA, "scratch_init")
        last = sc[sc.layer == sc.layer.max()].set_index(["activation", "init"])
        parts.append(
            f"At layer {int(sc.layer.max())}: tanh with unit-variance weights has "
            f"{_fmt(last.loc[('tanh', 'naive'), 'saturated_frac'])} of its units "
            f"saturated, against {_fmt(last.loc[('tanh', 'xavier'), 'saturated_frac'])} "
            f"with Xavier (activation sd {_fmt(last.loc[('tanh', 'xavier'), 'std'])}, "
            "a slow decay); ReLU's activation sd is "
            f"{_fmt(last.loc[('relu', 'he'), 'std'])} with He, "
            f"{_fmt(last.loc[('relu', 'xavier'), 'std'])} with Xavier (vanishing) and "
            f"{_fmt(last.loc[('relu', 'naive'), 'std'])} with unit variance (exploding).\n"
        )
        parts.append(
            _fig("01_scratch_optimizers_gradcheck")
            + "\n\n"
            + _fig("02_init_scan")
            + "\n"
        )
    except FileNotFoundError:
        pass

    # -- optimizers
    try:
        rep = load_table(AREA, "optimizers")
        chosen = load_table(AREA, "lr_chosen")
        s = summarise(rep)
        from physprior.optim.optimizers_study import BUDGET

        parts.append(
            _OPT_TEXT.replace(
                "{budgets}", ", ".join(f"{k} {v}" for k, v in BUDGET.items())
            )
        )
        parts.append(
            "Median over the reporting seeds 11/23/42 at the learning rate chosen on "
            "the tuning seeds. `evals to tol` is the number of loss-and-gradient "
            "evaluations until the data loss first reached the task's tolerance "
            "(— = never); `failure` is the fraction of seeds that never reached "
            "it or diverged or ended more than 10x above it; `param err` is the error in the key constant "
            "(R, gamma, D).\n"
        )
        for t in [t for t in TASK_LABEL if t in set(s.task)]:
            d = s[s.task == t].copy()
            d["order"] = d.model.map(
                {m: i for i, m in enumerate(MODEL_LABEL)}
            ) * 10 + d.optimizer.map({o: i for i, o in enumerate(OPT_ORDER)})
            d = d.sort_values("order")
            tol = rep[rep.task == t].tol.iloc[0]
            parts.append(
                f"\n**{TASK_LABEL[t]}** (tolerance {_fmt(tol)} in units of sd_y^2)\n"
            )
            parts.append(
                _table(
                    d,
                    [
                        "model",
                        "optimizer",
                        "lr",
                        "n_params",
                        "final_data_loss",
                        "evals_to_tol",
                        "failure_rate",
                        "nrmse_in",
                        "nrmse_out",
                        "param_err_pct",
                    ],
                    [
                        "model",
                        "optimizer",
                        "lr",
                        "params",
                        "data loss",
                        "evals to tol",
                        "failure",
                        "nrmse in",
                        "nrmse out",
                        "param err %",
                    ],
                )
                + "\n"
            )
        parts.append(_fig("03_lr_sensitivity") + "\n\n" + _fig("04_loss_curves") + "\n")
        g = load_table(AREA, "lr_grid")
        fr = (
            g.assign(fail=lambda x: ~x.reached_tol)
            .groupby(["model", "optimizer"])
            .fail.mean()
            .unstack()
            .reindex(
                index=[m for m in MODEL_LABEL if m in set(g.model)],
                columns=[o for o in OPT_ORDER if o in set(g.optimizer)],
            )
        )
        parts.append(
            "\nFailure rate over the whole learning-rate grid on the tuning seeds, all tasks "
            "pooled (how forgiving each method is of a badly chosen rate):\n\n"
            + _table(fr.reset_index(), ["model", *fr.columns])
            + "\n"
        )
        parts.append(_opt_findings(s, fr))
        n_sel = len(chosen)
        parts.append(
            f"\n{n_sel} learning rates were selected, all on the tuning seeds (`results/optim/lr_chosen.csv`).\n"
        )
    except FileNotFoundError:
        pass

    # -- curvature
    try:
        cv = load_table(AREA, "curvature")
        parts.append(_CURV_TEXT)
        agg = curvature_table(cv, load_json(AREA, "hessian_eigenvalues"))
        parts.append(
            _table(
                agg,
                [
                    "task",
                    "model",
                    "optimizer",
                    "n_params",
                    "lambda_max",
                    "n_sharp",
                    "frac_flat",
                    "n_negative",
                    "cond",
                    "phys_block_cond",
                    "lmax_data",
                    "lmax_phys",
                    "phys_over_data",
                ],
                [
                    "task",
                    "model",
                    "at",
                    "params",
                    "lambda_max",
                    "eig. > 1e-3 lambda_max",
                    "frac. abs(eig) < 1e-6 lambda_max",
                    "negative eig.",
                    "cond. (all)",
                    "cond. of constants block",
                    "lambda_max data (net)",
                    "lambda_max physics (net)",
                    "physics / data",
                ],
            )
            + "\n\n"
            + _fig("05_hessian_spectra")
            + "\n"
            + _curv_findings(agg)
        )
        c = load_table(AREA, "curves")
        d = c[(c.model == "pinn") & (c.optimizer == "adam")].dropna(
            subset=["grad_data", "grad_phys"]
        )
        if len(d):
            d = d.assign(ratio=d.grad_phys / d.grad_data)
            rows = []
            for t, e in d.groupby("task"):
                last = e[e.evals >= 0.8 * e.evals.max()]
                first = e[e.evals <= 0.1 * e.evals.max()]
                rows.append(
                    {
                        "task": t,
                        "ratio_early": float(first.ratio.median()),
                        "ratio_late": float(last.ratio.median()),
                        "ratio_max": float(e.ratio.max()),
                    }
                )
            parts.append(
                "\nGradient-norm ratio ||grad w L_phys|| / ||grad L_data|| over the "
                "network weights, PINN with Adam, reporting seeds (median over the first "
                "10% and the last 20% of training):\n\n"
                + _table(
                    pd.DataFrame(rows),
                    ["task", "ratio_early", "ratio_late", "ratio_max"],
                    ["task", "early", "late", "max"],
                )
                + "\n\n"
                + _fig("06_grad_ratio")
                + "\n"
            )
    except FileNotFoundError:
        pass

    # -- losses
    try:
        ls = load_table(AREA, "losses_summary")
        parts.append(_LOSS_TEXT)
        p = ls[ls.model == "physics"]
        parts.append(
            "**physics** (gamma bias = mean signed error over "
            f"{int(p.n.iloc[0])} fits per cell; z = bias / standard error)\n\n"
            + _table(
                p,
                [
                    "noise",
                    "loss",
                    "gamma_bias_pct",
                    "gamma_sd_pct",
                    "gamma_bias_z",
                    "nrmse_in",
                    "nrmse_out",
                ],
                [
                    "noise",
                    "loss",
                    "gamma bias %",
                    "gamma sd %",
                    "z",
                    "nrmse in",
                    "nrmse out",
                ],
            )
            + "\n\n**black-box NN** (median over the reporting seeds)\n\n"
            + _table(ls[ls.model == "nn"], ["noise", "loss", "nrmse_in", "nrmse_out"])
            + "\n\n"
            + _fig("07_loss_functions")
            + "\n"
        )
        parts.append(_loss_findings(ls))
        wd = load_table(AREA, "w_phys_dial")
        m = wd.groupby(["noise", "w_phys"]).median(numeric_only=True).reset_index()
        parts.append(
            _WDIAL_TEXT
            + f"Adam, rate {_fmt(wd.adam_lr.iloc[0])} (chosen for this PINN on the tuning seeds), "
            "3000 steps, cosine schedule. Medians over the reporting seeds.\n\n"
            + _table(
                m,
                [
                    "noise",
                    "w_phys",
                    "data_loss",
                    "phys_loss",
                    "nrmse_in",
                    "nrmse_out",
                    "err_gamma_pct",
                    "err_omega0_pct",
                ],
                [
                    "noise",
                    "w_phys",
                    "data loss",
                    "residual",
                    "nrmse in",
                    "nrmse out",
                    "gamma err %",
                    "omega0 err %",
                ],
            )
            + "\n\n"
            + _fig("08_w_phys_dial")
            + "\n"
            + _wdial_findings(m)
        )
    except FileNotFoundError:
        pass

    # -- balance
    try:
        bs = load_table(AREA, "balance_summary")
        fb = load_table(AREA, "balance_feedback")
        bf = load_table(AREA, "balance_fits")
        parts.append(_BAL_TEXT)
        parts.append(
            _table(
                bs,
                [
                    "track",
                    "split",
                    "variant",
                    "w_phys_final",
                    "correction_rms_frac",
                    "correction_rms_frac_out",
                    "nrmse_in",
                    "nrmse_out",
                    "nrmse_out_vs_physics",
                    "rms_gap_to_physics_out",
                ],
                [
                    "track",
                    "split",
                    "variant",
                    "w_phys final",
                    "corr. RMS (train)",
                    "corr. RMS (held out)",
                    "nrmse in",
                    "nrmse out",
                    "out / physics",
                    "gap to physics (out)",
                ],
            )
            + "\n"
        )
        b = bf[bf.variant == "balanced"]
        wmax = b.loc[b.w_phys_final.idxmax()]
        parts.append(
            f"\nLargest annealed weight: {_fmt(wmax.w_phys_final)} ({wmax.track}, "
            f"{wmax.split}, seed {int(wmax.seed)}). "
        )
        if len(fb):
            parts.append(
                f"The weight is still rising at the end of training in "
                f"{int((fb.w_growth_second_half > 1).sum())} of {len(fb)} balanced runs "
                f"(median growth over the second half: "
                f"{_fmt(fb.w_growth_second_half.median())}x); it has not settled. "
                "The pure feedback w ~ 1/|NN| would give a log-log slope of -1 between w "
                "and the correction's RMS; the measured slope over the second half has "
                f"median {_fmt(fb.slope_logw_vs_logcorr.median())} "
                f"(range {_fmt(fb.slope_logw_vs_logcorr.min())} to "
                f"{_fmt(fb.slope_logw_vs_logcorr.max())}): shallower, because the data "
                "term's gradient also shrinks as the fit converges.\n"
            )
        # Per-track verdict, computed.
        piv = bs.pivot_table(
            index=["track", "split"], columns="variant", values="nrmse_out"
        )
        lines = []
        for (tr, sp), r in piv.iterrows():
            lines.append(
                f"| {tr} | {sp} | {_fmt(r.get('balanced') / r.get('unbalanced'))} | "
                f"{_fmt(r.get('balanced') / r.get('physics'))} |"
            )
        parts.append(
            "\nOut-of-range error ratios (median over tuning seeds; < 1 means the "
            "first variant is better):\n\n| track | split | balanced / unbalanced | "
            "balanced / physics |\n|---|---|---|---|\n"
            + "\n".join(lines)
            + "\n\n"
            + _fig("09_balance_history")
            + "\n\n"
            + _fig("10_balance_errors")
            + "\n"
        )
        parts.append(_balance_implication(bs, bf))
    except FileNotFoundError:
        pass

    parts.append(_REPRO)
    text = "\n".join(parts)
    from pathlib import Path

    out = Path(path) if path else settings.root / "docs" / "optimization" / "README.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return text


# ---------------------------------------------------------------------------
# prose (method and theory only -- every number above is read from results)
# ---------------------------------------------------------------------------

_HEADER = """# Optimization: physics-constrained models versus black boxes

<!-- Generated by physprior.optim.report.render_doc() from results/optim.
     Do not edit by hand. -->

How optimizers, learning rates and loss functions behave for three model
families on the same problems:

- `physics`: the closed-form law with 1-3 constants trainable;
- `pinn`: on hydrogen the repo's shape B, `law(x; theta) + sd_y NN(x)` with
  the correction penalised (`w_phys = 1`, fixed); on the oscillator and the
  heat equation the residual PINN, where the network is the solution and the
  ODE/PDE residual is the physics term, constants trainable;
- `nn`: a tanh MLP, width 32, depth 3 (the repo's default arm).

Problems: `hydrogen` (real NIST H I levels, trained on n <= 15 minus five
held-out levels, out of range n = 16..40), `oscillator` (simulated
x'' + 2 gamma x' + omega0^2 x = 0, gamma and omega0 unknown, trained on
t in [0, 6], out of range t in (6, 10]), `heat` (simulated u_t = D u_xx,
D and two mode amplitudes unknown, trained on t in [0, 0.5], out of range
t in (0.5, 1]). Truth for the simulations is the analytic solution.

Code: `src/physprior/optim/`. Seeds: learning rates chosen on 3/7/19,
reported numbers on 11/23/42; the `balance` diagnostic uses 3/7/19 only, as
a diagnostic.
"""

_SCRATCH_TEXT = """
## 1. A network from scratch

`optim/scratch.py` is a NumPy MLP with the backward pass written out, and
SGD, heavy-ball momentum, Nesterov, RMSprop and Adam (with bias correction)
written by hand. It is teaching code; notebook T8 walks through it.

Initialisation: Xavier (variance 2/(fan_in + fan_out)) for tanh, He
(2/fan_in) for ReLU, because ReLU discards half the signal. The scan below
pushes random inputs through eight layers of width 64 and records the
spread of the activations and, for tanh, the fraction of units in the flat
tails (|a| > 0.99, where the derivative is below 0.02).

PINNs use tanh because their loss differentiates the network with respect to
its input. A ReLU network's second derivative is zero almost everywhere, so
an ODE residual containing u'' cannot be represented.
"""

_OPT_TEXT = """
## 2. Optimizers x learning rates x models

Optimizers: SGD, SGD with momentum 0.9, RMSprop, Adam, L-BFGS (strong-Wolfe
line search, every line-search evaluation charged to the budget), and for
`physics` also MINPACK Levenberg-Marquardt (`scipy.optimize.least_squares`,
same parameterisation and starting point; its finite-difference Jacobian
evaluations are counted). Learning rates: 1e-3 to 1 for the first-order
methods, 0.1 and 1 for L-BFGS. Constant rates, no schedule, so the optimizer
is what is measured. Budgets in evaluations: {budgets}. The seed sets the network initialisation and moves the starting
constants by ~30% in log space.
"""

_CURV_TEXT = """
## 3. Curvature at the solution

Hessian of the training loss with respect to every trainable parameter, by
double backpropagation (Hessian-vector products), at the end of the Adam and
L-BFGS runs on the reporting seeds. For `physics`, `nn` and the shape-B
PINN the full matrix is formed and diagonalised (at most a few thousand
parameters). For the residual PINNs, whose Hessian-vector product
differentiates the network three times, only the top 12 eigenvalues are
computed, by Lanczos (`scipy.sparse.linalg.eigsh` on the HVP operator), and
the columns that need the whole spectrum are left empty. The per-term top
eigenvalues also use Lanczos. A network's Hessian is singular, so its
condition number is not reported; instead the number of sharp directions
(eigenvalues above 1e-3 lambda_max), the fraction of flat ones (|lambda|
below 1e-6 lambda_max) and the number of negative ones. `cond. of constants block` is the
condition number of the sub-Hessian over the physical constants alone.
`physics / data` compares the sharpest curvature each loss term induces on
the network weights of a PINN.

What the theory predicts: the `physics` loss is low-dimensional and, in the
log parameterisation, reasonably conditioned, so Gauss-Newton methods (LM,
and L-BFGS which approximates them) converge in tens of evaluations. The
black box has a spectrum that is mostly flat with a few sharp directions:
the sharpest sets the largest stable step (lr < 2 / lambda_max for SGD) and
the flat ones make progress slow and leave the solution undetermined along
them. The residual PINN's physics term differentiates the network, which
multiplies each Fourier component of the error by its frequency (twice for
a second derivative), so its curvature on the network weights exceeds the
data term's and the gradients of the two terms are unbalanced (Wang, Teng &
Perdikaris 2021; Krishnapriyan et al. 2021).
"""

_LOSS_TEXT = """
## 4. Loss functions

`optim/losses_study.py`. The oscillator with gamma and omega0 unknown, 40
training points, three noise models: Gaussian (sd 0.05), Student-t with
nu = 2 (scale 0.05, infinite variance), and Gaussian plus 10% outliers
displaced by 0.5-1.0. Losses: MSE, L1, Huber (delta = 1.345 s), Cauchy
(c = 2.385 s), with s = 1.4826 MAD of an MSE fit's residuals (the standard
95%-efficiency constants, not tuned), and the Gaussian NLL with a learned
sigma (log sigma linear in t for `physics`, a second network output for
`nn`). `physics` starts within ~10% of the truth, so the table measures the
estimator, not the optimizer. Reporting seeds, four noise draws each for
`physics`.
"""

_WDIAL_TEXT = """
### The physics weight as a loss hyperparameter

The residual PINN on the oscillator at w_phys from 0 to 100, Gaussian noise
and outliers. At w_phys = 0 the constants receive no gradient at all (they
appear only in the residual), so their "error" there is the starting guess.
"""

_BAL_TEXT = """
## 5. The frozen `balance` option drives the physics weight up

`FROZEN_PINN = PinnOptions(balance=True)` anneals the physics weight every
100 epochs toward max|grad L_data| / mean|grad L_phys| over the correction
network's weights (`methods/pinn.py::_annealed_weight`). For shape B,
L_phys = mean(NN^2), so grad L_phys = 2 mean(NN grad NN) is proportional to
the correction itself. A larger weight shrinks the correction, which shrinks
the physics gradient, which raises the next weight: the annealing has no
fixed point short of NN = 0, and w keeps growing for as long as training
runs. (In Wang et al. the physics term is a PDE residual, whose gradient does
not vanish with the network's output.)

Measured with `fit_pinn` exactly as `fit_arm` calls it (6000 epochs, Adam,
cosine schedule, `w_phys = 1` start), on the four real tracks, both the
track's random interpolation split and its extrapolation split, tuning seeds
3/7/19. `unbalanced` is the same arm with `balance` off (`w_phys = 1`
fixed). `corr. RMS` is the correction's RMS in units of the training sd_y,
on the training points (`Fit.extra["correction_rms_frac"]`) and on the
held-out points (prediction minus the law at the PINN's own constants).
`gap to physics` is the RMS difference between the PINN's and the `physics`
fit's held-out predictions, in units of the spread of y.
"""


def _opt_findings(s: pd.DataFrame, fr: pd.DataFrame) -> str:
    """Which optimizer reaches the tolerance first, per task and model."""
    lines = [
        "\nFastest optimizer to the tolerance (median evaluations, reporting seeds):\n"
    ]
    lines.append("| task | model | fastest | evals | Adam evals | L-BFGS evals |")
    lines.append("|---|---|---|---|---|---|")
    for t in [t for t in TASK_LABEL if t in set(s.task)]:
        for m in [m for m in MODEL_LABEL if m in set(s.model)]:
            d = s[(s.task == t) & (s.model == m)].set_index("optimizer")
            ok = d[d.failure_rate < 1].dropna(subset=["evals_to_tol"])
            if not len(ok):
                lines.append(f"| {t} | {m} | none reached it | — | — | — |")
                continue
            best = ok.evals_to_tol.idxmin()
            lines.append(
                f"| {t} | {m} | {best} | {_fmt(ok.evals_to_tol.min())} | "
                f"{_fmt(d.evals_to_tol.get('adam', np.nan))} | "
                f"{_fmt(d.evals_to_tol.get('lbfgs', np.nan))} |"
            )
    worst_first = fr.drop(columns=[c for c in ("lbfgs",) if c in fr.columns])
    phys = s[(s.model == "physics") & (s.failure_rate < 1)].dropna(
        subset=["evals_to_tol"]
    )
    phys_best = phys.loc[phys.groupby("task").evals_to_tol.idxmin()]
    second = set(phys_best.optimizer) <= {"lbfgs", "lm"}
    least = worst_first.mean().idxmin()
    res = s[(s.model == "pinn") & s.task.isin(["oscillator", "heat"])]
    ok_res = sorted(set(res[res.failure_rate < 1].optimizer))
    lines.append(
        "\nFor `physics` the fastest method on every task is "
        + ("second-order (L-BFGS or LM)" if second else "not always second-order")
        + ": "
        + ", ".join(
            f"{t} {_fmt(v)}"
            for t, v in zip(phys_best.task, phys_best.evals_to_tol, strict=True)
        )
        + " evaluations to tolerance. Over the whole learning-rate grid L-BFGS fails least "
        "for every family ("
        + ", ".join(f"{m} {_fmt(fr.loc[m, 'lbfgs'])}" for m in fr.index)
        + f"; first-order methods {_fmt(worst_first.min().min())} to "
        f"{_fmt(worst_first.max().max())}), and among first-order methods {least} "
        "is the least sensitive to the rate. On the residual PINNs the methods "
        f"that reach the tolerance on at least one seed are: {', '.join(ok_res)}.\n"
    )
    return "\n".join(lines)


def curvature_table(cv: pd.DataFrame, eig: list[dict]) -> pd.DataFrame:
    """Median spectrum statistics per (task, model, optimizer).

    From the stored eigenvalues where the whole spectrum is known: how many
    directions are sharp (above 1e-3 lambda_max), what fraction is flat
    (|lambda| below 1e-6 lambda_max, i.e. zero to the precision that
    matters), how many are negative (not a minimum), and the plain condition
    number lambda_max / lambda_min, which is only meaningful for a
    positive-definite Hessian.
    """
    rows = []
    method = dict(
        zip(
            zip(cv.task, cv.model, cv.optimizer, cv.seed, strict=True),
            cv.method,
            strict=True,
        )
    )
    for e in eig:
        v = np.sort(np.asarray(e["eigenvalues"], float))[::-1]
        full = method.get((e["task"], e["model"], e["optimizer"], e["seed"])) == "full"
        lmax = v[0]
        rows.append(
            {
                "task": e["task"],
                "model": e["model"],
                "optimizer": e["optimizer"],
                "seed": e["seed"],
                "n_sharp": int(np.sum(v > 1e-3 * lmax)) if full else np.nan,
                "frac_flat": float(np.mean(np.abs(v) < 1e-6 * lmax))
                if full
                else np.nan,
                "n_negative": int(np.sum(v < -1e-6 * lmax)) if full else np.nan,
                "cond": float(lmax / v[-1]) if full and v[-1] > 0 else np.nan,
            }
        )
    spec = pd.DataFrame(rows)
    base = cv[
        [
            "task",
            "model",
            "optimizer",
            "seed",
            "n_params",
            "lambda_max",
            "phys_block_cond",
            "lambda_max_data_net",
            "lambda_max_phys_net",
        ]
    ].merge(spec, on=["task", "model", "optimizer", "seed"])
    agg = (
        base.groupby(["task", "model", "optimizer"])
        .agg(
            n_params=("n_params", "first"),
            lambda_max=("lambda_max", "median"),
            n_sharp=("n_sharp", "median"),
            frac_flat=("frac_flat", "median"),
            n_negative=("n_negative", "median"),
            cond=("cond", "median"),
            phys_block_cond=("phys_block_cond", "median"),
            lmax_data=("lambda_max_data_net", "median"),
            lmax_phys=("lambda_max_phys_net", "median"),
        )
        .reset_index()
    )
    agg["phys_over_data"] = agg.lmax_phys / agg.lmax_data
    return agg


def _curv_findings(agg: pd.DataFrame) -> str:
    a = agg[agg.optimizer == "lbfgs"].set_index(["task", "model"])
    phys = a.xs("physics", level="model")
    nn = a.xs("nn", level="model")
    pinn = a.xs("pinn", level="model")
    out = [
        "\nFindings, computed from the table (at the L-BFGS solutions). The "
        "`physics` Hessians are positive definite with condition numbers "
        + ", ".join(f"{t} {_fmt(r.cond)}" for t, r in phys.iterrows())
        + ". The black box's spectrum has "
        + ", ".join(
            f"{_fmt(r.n_sharp)} sharp directions of {int(r.n_params)} ({t})"
            for t, r in nn.iterrows()
        )
        + "; the rest are flat or nearly so (flat fraction "
        + ", ".join(f"{t} {_fmt(r.frac_flat)}" for t, r in nn.iterrows())
        + "). Negative eigenvalues at the Adam end points (see the `adam` rows) "
        "show that a constant-rate Adam run stops short of a minimum; L-BFGS "
        "ends closer to one."
    ]
    if "hydrogen" in pinn.index:
        r = pinn.loc["hydrogen"]
        out.append(
            f" For the shape-B PINN on hydrogen the two terms induce the same "
            f"sharpest curvature on the network (ratio {_fmt(r.phys_over_data)}): "
            "near a solution both are Gauss-Newton forms J^T J of the same network "
            "Jacobian, so shape B has no stiffness imbalance for loss balancing to "
            "correct."
        )
    for t in ("oscillator", "heat"):
        if t in pinn.index:
            r = pinn.loc[t]
            out.append(
                f" Residual PINN, {t}: physics/data curvature ratio "
                f"{_fmt(r.phys_over_data)}."
            )
    ratios = {
        t: pinn.loc[t].phys_over_data for t in ("oscillator", "heat") if t in pinn.index
    }
    over = [t for t, v in ratios.items() if v > 1]
    under = [t for t, v in ratios.items() if v <= 1]
    if over or under:
        out.append(
            " The prediction that the residual term is the stiffer one holds on "
            + (", ".join(over) if over else "neither task")
            + (
                f" and does not hold on {', '.join(under)}, where the diffusivity "
                "(D = 0.1) scales the second-derivative term down"
                if "heat" in under
                else (f" and not on {', '.join(under)}" if under else "")
            )
            + "."
        )
    out.append(
        " The cond. of the constants block is infinite for the heat PINN because "
        "the two mode amplitudes do not enter the residual loss at all (only D "
        "does); they are fitted by nothing and their values there are the "
        "starting guess.\n"
    )
    return "".join(out)


def _wdial_findings(m: pd.DataFrame) -> str:
    out = ["\nFindings, computed from the table. "]
    for nz, d in m.groupby("noise"):
        d = d.set_index("w_phys")
        wo = d.nrmse_out.idxmin()
        wg = d.err_gamma_pct.abs().idxmin()
        top = d.index.max()
        out.append(
            f"With {nz} noise the out-of-range error is lowest at w_phys = {_fmt(wo)} "
            f"({_fmt(d.nrmse_out[wo])}, against {_fmt(d.nrmse_out[0.0])} at w = 0) "
            f"and the damping error at w_phys = {_fmt(wg)} ({_fmt(d.err_gamma_pct[wg])}%). "
            f"At w_phys = {_fmt(top)} the residual is smallest "
            f"({_fmt(d.phys_loss[top])}) but the data loss rises to "
            f"{_fmt(d.data_loss[top])} and gamma is off by "
            f"{_fmt(d.err_gamma_pct[top])}%. "
        )
    out.append(
        "The curve is a trade-off with an interior optimum (measured on the "
        "reporting seeds; nothing is selected from it): too little weight "
        "and the constants are unconstrained, too much and the optimizer "
        "satisfies the ODE with a solution that ignores the data, which the "
        "initial conditions alone permit for any damping.\n"
    )
    return "".join(out)


def _loss_findings(ls: pd.DataFrame) -> str:
    """The loss-function comparisons, computed from the summary."""
    p = ls[ls.model == "physics"].set_index(["noise", "loss"])
    n = ls[ls.model == "nn"].set_index(["noise", "loss"])
    zmax = float(p.gamma_bias_z.abs().max())
    zarg = p.gamma_bias_z.abs().idxmax()
    o = p.loc["outliers"]
    best_o = o.nrmse_in.idxmin()
    no = n.loc["outliers"]
    best_n = no.nrmse_in.idxmin()
    return (
        "\nFindings, computed from the table. "
        + (
            "No loss gives a gamma bias beyond three standard errors at these "
            "sample sizes"
            if zmax < 3
            else "At least one loss gives a gamma bias beyond three standard errors"
        )
        + f": the largest |z| is {_fmt(zmax)} ({zarg[0]}, {zarg[1]}). All losses in "
        "a row are fitted to the same noise draws, so their biases are correlated "
        "and an offset common to a row comes from the draws, not the losses. "
        "What outliers change is the spread: "
        f"the scatter of gamma under MSE is {_fmt(o.loc['mse', 'gamma_sd_pct'])}% "
        f"against {_fmt(o.loc['huber', 'gamma_sd_pct'])}% (Huber) and "
        f"{_fmt(o.loc['cauchy', 'gamma_sd_pct'])}% (Cauchy), and the in-range "
        f"prediction error is {_fmt(o.loc['mse', 'nrmse_in'])} (MSE) against "
        f"{_fmt(o.loc[best_o, 'nrmse_in'])} ({best_o}). The Gaussian NLL with a "
        "learned sigma does not help against outliers "
        f"({_fmt(o.loc['gauss_nll', 'nrmse_in'])}). The likely reason: a sigma smooth in t "
        "cannot single out isolated points, so it widens everywhere. For the black "
        f"box the best in-range loss under outliers is {best_n} "
        f"({_fmt(no.loc[best_n, 'nrmse_in'])} against "
        f"{_fmt(no.loc['mse', 'nrmse_in'])} for MSE), but out of range every "
        f"loss leaves it at an nrmse between {_fmt(n.nrmse_out.min())} and "
        f"{_fmt(n.nrmse_out.max())}, against at most "
        f"{_fmt(p.nrmse_out.max())} for `physics`: the loss function changes "
        "what the network fits, not what it knows outside the data.\n"
    )


def _balance_implication(bs: pd.DataFrame, bf: pd.DataFrame) -> str:
    """The Phase 2 implication, with its numbers computed from the summary."""
    piv = bs.pivot_table(
        index=["track", "split"], columns="variant", values="nrmse_out"
    )
    corr = bs[bs.variant == "balanced"].set_index(["track", "split"])
    n = len(piv)
    vs_phys = piv["balanced"] / piv["physics"]
    vs_unb = piv["balanced"] / piv["unbalanced"]
    near = vs_phys[(vs_phys > 0.95) & (vs_phys < 1.05)]
    worse = vs_phys[vs_phys >= 1.05]
    better = vs_phys[vs_phys <= 0.95]
    w_med = corr.w_phys_final.median()
    c_max = corr.correction_rms_frac.max()
    bal = bf[bf.variant == "balanced"]
    devs = [c for c in bal.columns if c.startswith("dev_") and c.endswith("_ppm")]
    dev_max = float(np.nanmax(np.abs(bal[devs].to_numpy()))) if devs else np.nan
    lines = [
        "\n### What this implies for the Phase 2 conclusion\n",
        f"Across the {n} track/split cells the median annealed weight at the end of "
        f"training is {_fmt(w_med)}, and the balanced correction's RMS on the "
        f"training points is at most {_fmt(c_max)} of sd_y. Out of range the "
        f"balanced PINN beats the unbalanced one in {int((vs_unb < 1).sum())} of {n} "
        f"cells (median ratio {_fmt(vs_unb.median())}). Against the `physics` fit it "
        f"is within 5% in {len(near)} cells, worse by 5% or more in {len(worse)}"
        + (
            f" ({', '.join(f'{t} {sp}: {_fmt(v)}x' for (t, sp), v in worse.items())})"
            if len(worse)
            else ""
        )
        + f", and better by 5% or more in {len(better)}"
        + (
            f" ({', '.join(f'{t} {sp}: {_fmt(v)}x' for (t, sp), v in better.items())})"
            if len(better)
            else ""
        )
        + ". The balanced PINN's constants agree with the `physics` fit's to within "
        f"{_fmt(dev_max)} ppm in every run."
        + (
            " In no cell does it beat `physics` out of range by more than 5%."
            if not len(better)
            else ""
        )
        + "\n",
        """
So `balance` does not make the correction network useful; it switches it
off on the training points. The Phase 2 improvement is measured against the
unbalanced PINN, whose correction overfits and extrapolates badly, and the
balanced PINN wins that comparison by becoming the `physics` fit with a
residual network attached. Where it is worse, the constants are not the cause (they
match to the ppm figure above); the cause is what remains of the
correction. It is small in units of sd_y, but not small next to the `physics`
fit's own error on these tracks (hydrogen's is ~1e-6 of the spread), and on
the extrapolation splits it is larger on the held-out points than on the
training points (compare the two correction columns in the table).

Consequences for reading the results tables: every `pinn` number produced
with `FROZEN_PINN` on an algebraic track is, to within these ratios, a
second copy of `physics`. "The PINN ties `physics`" is then expected by
construction and is not evidence that a learned correction is harmless or
helpful. On helium, where the law is incomplete and hypothesis H3 asks what a
correction learns, the frozen arm cannot answer, which the helium track
already works around with its `pinn_unbalanced` diagnostic. The frozen
configuration is not changed here. Re-deciding it is a tuning-seed decision;
the options are to keep it and describe `pinn` as a regularised `physics`
fit, to cap the annealed weight, or to report `pinn` without `balance`.
""",
    ]
    return "\n".join(lines)


_REPRO = """
## Reproduce

```bash
python -c "from physprior.optim import report; report.run_all()"   # full study
python -c "from physprior.optim import report; report.make_figures(); report.render_doc()"
```

Outputs: `results/optim/*.csv|json`, `figures/optim/*.png`, this file.
"""


def dump_summary() -> str:
    """A compact JSON of the headline comparisons, for quick inspection."""
    rep = load_table(AREA, "optimizers")
    return json.dumps(optimizer_findings(rep), indent=2, default=str)
