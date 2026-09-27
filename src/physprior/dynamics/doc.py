"""docs/dynamics/README.md, generated from results/dynamics/.

Every number on the page is read from a results file; the prose describes
method only. Regenerate with `render_doc()` after `study.run()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from physprior.config import get_settings
from physprior.io import load_json

from . import TRACK
from . import pdes as PD
from . import systems as S
from .figures import ARM_LABEL
from .study import load_rows, summary_table

DOC = "dynamics/README.md"

STATEMENTS = {
    "E1": "Leapfrog HNN keeps energy error bounded; black boxes drift",
    "E2": "A Hamiltonian field under RK4 still drifts more than under leapfrog",
    "E3": "Residual u + dt NN(u) beats direct u' = NN(u)",
    "E4": "Neural ODE (trained through RK4) beats the residual stepper",
    "E5": "Known physics + closure beats the black boxes",
    "E6": "An unrolled loss lengthens the valid horizon",
    "E7": "Local conv beats dense MLP on PDEs, most with little data",
    "E8": "On Lorenz, one-step error orders the valid times",
    "E9": "Structure helps most with little data (pendulum)",
}


def _fmt(v, nd=3) -> str:
    if v is None:
        return "–"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        x = float(v)
        if not np.isfinite(x):
            return "inf" if x > 0 else "–"
        if x == 0:
            return "0"
        return f"{x:.{nd}g}"
    return str(v)


def _table(df: pd.DataFrame, nd: int = 3) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(_fmt(r[c], nd) for c in cols) + " |")
    return "\n".join(out)


def _fig(name: str, alt: str) -> str:
    return f"![{alt}](../../figures/{TRACK}/{name}.png)"


def _flat(d, prefix="") -> list[str]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out += _flat(v, f"{prefix}{k}.")
        else:
            out.append(f"{prefix}{k} = {_fmt(v)}")
    return out


# ---------------------------------------------------------------------------


def systems_table() -> pd.DataFrame:
    rows = []
    for name in S.SYSTEMS:
        s = S.get_system(name)
        rows.append(
            {
                "system": name,
                "law": s.description,
                "dt": s.dt,
                "train steps": s.train_steps,
                "test steps": s.test_steps,
                "structure": ", ".join(
                    x
                    for x, on in (
                        ("Hamiltonian", s.hamiltonian),
                        ("closure", s.known is not None),
                        ("chaotic", s.chaotic),
                    )
                    if on
                )
                or "–",
            }
        )
    for name, p in PD.PDES.items():
        rows.append(
            {
                "system": name,
                "law": p.description,
                "dt": p.dt,
                "train steps": p.train_steps,
                "test steps": p.test_steps,
                "structure": "PDE, 64-point periodic grid"
                + (", closure" if p.known_rhs is not None else ""),
            }
        )
    return pd.DataFrame(rows)


def convergence_table(conv: dict) -> pd.DataFrame:
    rows = []
    for c in conv["ode"]:
        orders = [r.get("observed_order") for r in c["rows"][1:]]
        rows.append(
            {
                "system": c["system"],
                "substeps used": c["used_substeps"],
                "max error (used vs 4x finer)": c["used_max_err"],
                "observed order": float(np.nanmedian(orders)),
            }
        )
    b = conv.get("burgers")
    if b:
        rows.append(
            {
                "system": "burgers (spectral)",
                "substeps used": b["used"]["substeps"],
                "max error (used vs 4x finer)": b["used_max_err"],
                "observed order": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def verdict_table(ver: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": k, "expectation": STATEMENTS[k], "verdict": v["verdict"]}
            for k, v in ver.items()
        ]
    )


def _label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["arm"] = [ARM_LABEL.get(a, a) for a in df["arm"]]
    return df


ODE_COLS = {
    "onestep": "one-step err",
    "err_train_h": "err @ train horizon",
    "err_end": "err @ end",
    "valid_over_train": "valid horizon / train length",
    "blowup_frac": "blow-up frac",
    "diverged_frac": "frac err > 1 @ end",
    "train_seconds": "train s",
    "n_params": "params",
}


def ode_table(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    t = summary_table(rows, "ode", split)
    keep = ["system", "arm"] + [c for c in ODE_COLS if c in t]
    return _label(t[keep].rename(columns=ODE_COLS))


def invariant_table(rows: pd.DataFrame) -> pd.DataFrame:
    t = summary_table(rows, "ode", "test")
    t = t[t.system.isin(["pendulum", "kepler"])]
    cols = [
        c
        for c in (
            "H_early",
            "H_late",
            "H_growth",
            "E_early",
            "E_late",
            "E_growth",
            "L_late",
            "L_growth",
        )
        if c in t
    ]
    return _label(t[["system", "arm", *cols]])


def lorenz_table(rows: pd.DataFrame) -> pd.DataFrame:
    t = summary_table(rows, "ode", "test")
    t = t[t.system == "lorenz"]
    cols = [c for c in ("onestep", "vpt_lyap", "blowup_frac") if c in t]
    return _label(
        t[["arm", *cols]].rename(columns={"vpt_lyap": "valid time [Lyapunov]"})
    )


def objective_table(rows: pd.DataFrame) -> pd.DataFrame:
    ob = rows[rows.split == "test"]
    out = []
    for (s, a), _ in ob[ob.exp == "objective"].groupby(["system", "arm"]):
        one = ob[(ob.system == s) & (ob.arm == a) & (ob.exp == "main")]
        unr = ob[(ob.system == s) & (ob.arm == a) & (ob.exp == "objective")]
        out.append(
            {
                "system": s,
                "arm": a,
                "valid/train, one-step loss": one.valid_over_train.median(),
                "valid/train, unrolled loss": unr.valid_over_train.median(),
                "err @ end, one-step": one.err_end.median(),
                "err @ end, unrolled": unr.err_end.median(),
                "train s, one-step": one.train_seconds.median(),
                "train s, unrolled": unr.train_seconds.median(),
            }
        )
    return _label(pd.DataFrame(out)) if out else pd.DataFrame()


def data_table(rows: pd.DataFrame) -> pd.DataFrame:
    d = rows[
        (rows.split == "test")
        & rows.exp.isin(["main", "data"])
        & rows.system.isin(["pendulum", "burgers"])
    ]
    col = {"pendulum": "valid_over_train", "burgers": "err_end"}
    out = []
    for (s, a), g in d.groupby(["system", "arm"]):
        r = {"system": s, "arm": a, "metric": col[s]}
        for n, gg in g.groupby("n_traj"):
            r[f"{int(n)} traj"] = gg[col[s]].median()
        out.append(r)
    return _label(pd.DataFrame(out))


PDE_COLS = {
    "onestep": "one-step err",
    "err_train_h": "err @ train horizon",
    "err_end": "err @ end",
    "valid_over_train": "valid horizon / train length",
    "blowup_frac": "blow-up frac",
    "diverged_frac": "frac err > 1 @ end",
    "mass_drift_end": "mass drift",
    "energy_ratio_end": "energy ratio @ end",
    "train_seconds": "train s",
    "n_params": "params",
}


def pde_table(rows: pd.DataFrame, split: str) -> pd.DataFrame:
    t = summary_table(rows, "pde", split)
    keep = ["system", "arm"] + [c for c in PDE_COLS if c in t]
    return _label(t[keep].rename(columns=PDE_COLS))


# ---------------------------------------------------------------------------


def _caps() -> dict[str, float]:
    caps = {
        n: S.get_system(n).test_steps / S.get_system(n).train_steps for n in S.SYSTEMS
    }
    caps.update({n: p.test_steps / p.train_steps for n, p in PD.PDES.items()})
    return caps


def _grid1_section(ver: dict) -> list[str]:
    try:
        old = load_json(TRACK, "expectations_grid1")
        edge = load_json(TRACK, "tuning_edge_grid1")
    except FileNotFoundError:
        return []
    moved = sorted(k for k, v in edge["arms"].items() if v["would_change"])
    rows = [
        {
            "id": k,
            "verdict, first grid": old[k]["verdict"],
            "verdict, reported": v["verdict"],
        }
        for k, v in ver.items()
        if k in old
    ]
    return [
        "The first full run used the grid (0.001, 0.003). A check on the tuning "
        f"seeds at {_fmt(edge['edge_lr'])} would have moved the choice for "
        f"{len(moved)} of {len(edge['arms'])} arms ({', '.join(moved)}): the "
        "networks were under-trained at the fixed step budget. The grid was "
        "widened and the whole study rerun. The superseded verdicts are kept in "
        "`results/dynamics/expectations_grid1.json`:",
        "",
        _table(pd.DataFrame(rows)),
        "",
    ]


def _edge_section() -> list[str]:
    try:
        edge = load_json(TRACK, "tuning_edge")
    except FileNotFoundError:
        return []
    rows = [
        {
            "arm": k,
            "chosen lr": v["chosen"],
            "choice with lr " + _fmt(edge["edge_lr"]): v["with_edge"],
            **{
                f"val valid steps @ {lr}": n
                for lr, n in v["val_valid_steps_by_lr"].items()
            },
        }
        for k, v in edge["arms"].items()
    ]
    return [
        "Edge check: the tuning runs repeated at one larger rate than the "
        "grid (tuning seeds and validation set only). The table shows whether "
        "the choice would have moved; the reported runs keep the frozen choice.",
        "",
        _table(pd.DataFrame(rows)),
        "",
    ]


def render_doc(path=None) -> str:
    """Write the page (to docs/dynamics/README.md unless `path` is given)."""
    conv = load_json(TRACK, "convergence")
    ver = load_json(TRACK, "expectations")
    meta = load_json(TRACK, "meta")
    rows = load_rows()
    o = meta["options"]
    lr = meta.get("lr") or {}
    lyap = conv.get("lyapunov") or {}

    parts = [
        "# Learning the update rule of a dynamical system",
        "",
        "Generated by `physprior.dynamics.doc.render_doc()` from "
        "`results/dynamics/`. Do not edit by hand.",
        "",
        "A model sees pairs of states one observation interval apart, "
        "u(t) and u(t + dt), and learns the map F with u(t + dt) = F(u(t)). "
        "It is then iterated on its own output from true initial states, for "
        "five to ten times longer than any training trajectory. The question is "
        "what physical structure in the stepper buys over a black box on those "
        "long rollouts. The expectations and the outcomes that would refute "
        "them were written before the reported runs: "
        "[expectations.md](expectations.md).",
        "",
        "## Verdicts",
        "",
        _table(verdict_table(ver)),
        "",
        "The numbers behind each verdict, medians over the reporting seeds:",
        "",
    ]
    for k, v in ver.items():
        parts.append(f"- **{k}**: " + "; ".join(_flat(v["detail"])))
        for c in v.get("caveats", []):
            parts.append(f"  - {c}")
    parts += [
        "",
        "Indented lines are measurements the pre-registered criterion does "
        "not look at; they were added to the evaluation after the criteria "
        "were fixed and do not change a verdict.",
    ]
    parts += [
        "",
        "## Systems",
        "",
        _table(systems_table()),
        "",
        "Units are dimensionless (g/l = 1 for the pendulum, GM = 1 for Kepler). "
        "Training trajectories start from initial states drawn per seed; the "
        "test set and the out-of-distribution (OOD) set are fixed across arms "
        "and seeds. OOD means energies closer to the separatrix (pendulum), "
        "larger amplitudes (Duffing), eccentricities 0.5 to 0.6 against a "
        "training range 0 to 0.5 (Kepler), and rougher, larger initial fields "
        "(PDEs: modes up to k = 10 against 6). Lorenz-63 has no OOD set; its "
        "initial states are on the attractor.",
        "",
        "### Reference solutions and their convergence",
        "",
        "ODE references use RK4 at the listed substeps per observation "
        "interval (the pendulum and Kepler use the project's second-order "
        "`numerics.integrators.rk4`). The error is the maximum over one "
        "training trajectory against a run with four times the substeps, in "
        "training standard deviations. Burgers uses an integrating-factor "
        "pseudo-spectral solver on 1024 points, compared with 2048 points and "
        "half the time step at the coarse grid points over a full test rollout. "
        "Heat and advection are solved exactly in Fourier space.",
        "",
        _table(convergence_table(conv)),
        "",
    ]
    if lyap:
        parts += [
            f"Largest Lyapunov exponent of Lorenz-63 re-measured on the reference "
            f"integrator: {_fmt(lyap['measured'], 4)} against "
            f"{_fmt(lyap['literature'], 4)} (Sprott 2003); relative difference "
            f"{_fmt(lyap['rel_diff'], 2)}. Valid times are quoted in units of "
            "1/lambda_1 with the literature value.",
            "",
        ]
    parts += [
        "## Steppers and protocol",
        "",
        "| arm | update | structure it carries |",
        "|---|---|---|",
        "| direct | u' = NN(u) | none; must learn the identity |",
        "| residual | u' = u + dt NN(u) | increment form (Euler-like) |",
        "| neural ODE | u' = RK4(NN, u, dt) | vector field, trained through RK4 |",
        "| HNN (RK4) | u' = RK4(J grad H_NN, u, dt) | Hamiltonian field (Greydanus et al. 2019) |",
        "| HNN (leapfrog) | leapfrog on T_NN(p) + V_NN(q) | separable H, symplectic map (Chen et al. 2020) |",
        "| closure | u' = RK4(f_known + NN, u, dt) | known part of the law (as in APHYNITY, Yin et al. 2021) |",
        "| local conv | u' = u + dt CNN(u) | kernel-5 stencil, translation-equivariant |",
        "| dense MLP | u' = u + dt MLP(u) | none across space |",
        "| PDE closure | u' = u + dt (nu D2 u + CNN(u)) | viscous term known, Burgers only |",
        "",
        "Reference rows, not competitors: the true field with one RK4 step of "
        "size dt (`true field, RK4 at dt`), the incomplete known physics alone, "
        "and for PDEs the correct equation on the 64-point grid with "
        "second-order central differences and RK4 substeps (`correct PDE, "
        "coarse FD`).",
        "",
        "The closure systems are given: the small-angle pendulum q'' = -q; the "
        "undamped linear Duffing oscillator with its drive (damping and the "
        "cubic spring are learned); the linear part of Lorenz-63 (the products "
        "xz and xy are learned); and the viscous term of Burgers.",
        "",
        f"Every run uses {o['steps']} Adam steps with cosine decay, gradient "
        f"clipping at norm 1, batch {o['batch']} windows for ODEs and "
        f"{o.get('pde_batch', o['batch'])} fields for PDEs, {o['n_traj']} training "
        f"trajectories, networks of width 64 and two hidden layers (tanh) for "
        f"ODEs and 32 channels for the convolutional steppers. The learning "
        f"rate was chosen per arm from {o['lrs']} on the tuning seeds "
        f"{tuple(o['tune_seeds'])} by the median validation valid horizon, on "
        f"the pendulum for ODE arms and on Burgers for PDE arms, then frozen: "
        + ", ".join(f"{k} {_fmt(v)}" for k, v in sorted(lr.items()))
        + f". Reported numbers are medians over the reporting seeds "
        f"{tuple(o['seeds'])} of the median over {o['n_test']} test initial "
        f"states.",
        "",
        f"Errors are in training standard deviations per component (ODEs) or "
        f"in units of the initial RMS fluctuation of each field (PDEs). The "
        f"valid horizon is the number of steps before the error first exceeds "
        f"{meta['thresholds']['valid']} ({meta['thresholds']['lorenz']} for "
        f"Lorenz-63, the threshold of Pathak et al. 2018), divided here by the "
        f"length of a training trajectory; it is capped by the rollout length ("
        + ", ".join(f"{k} {v:g}" for k, v in _caps().items())
        + f"). A rollout that leaves the data by "
        f"more than {meta['thresholds']['blowup_z']} standard deviations or "
        f"produces a non-finite state counts as a blow-up.",
        "",
        *_grid1_section(ver),
        *_edge_section(),
        "## Rollout error",
        "",
        _fig("ode_error_vs_horizon", "ODE rollout error vs horizon"),
        "",
        "In distribution:",
        "",
        _table(ode_table(rows, "test")),
        "",
        "Out of distribution:",
        "",
        _fig("ode_error_vs_horizon_ood", "ODE rollout error vs horizon, OOD"),
        "",
        _table(ode_table(rows, "ood")),
        "",
        "## Conserved quantities",
        "",
        "`early` and `late` are the median relative invariant error over the "
        "first and last tenth of the rollout; `growth` is late / early. A "
        "bounded error has growth near 1.",
        "",
        _fig("invariants", "Energy and angular momentum drift"),
        "",
        _table(invariant_table(rows)),
        "",
        _fig("pendulum_phase", "Pendulum phase portraits"),
        "",
        "## Chaos: Lorenz-63",
        "",
        _fig("lorenz_valid_time", "Lorenz valid prediction time"),
        "",
        _table(lorenz_table(rows)),
        "",
        "## Training objective: one-step vs unrolled loss",
        "",
        f"Same architecture, learning rate and number of gradient steps; the "
        f"unrolled loss averages the error over {o['horizon']} steps of the "
        f"model's own rollout from each window start.",
        "",
        _fig("objective", "One-step vs unrolled training loss"),
        "",
        _table(objective_table(rows)),
        "",
        "## Data efficiency",
        "",
        "The number of gradient steps is the same at every data size.",
        "",
        _fig("data_efficiency", "Data efficiency"),
        "",
        _table(data_table(rows)),
        "",
        "## PDEs: locality and translation equivariance",
        "",
        _fig("pde_error_vs_horizon", "PDE rollout error vs horizon"),
        "",
        _table(pde_table(rows, "test")),
        "",
        "Out of distribution:",
        "",
        _fig("pde_error_vs_horizon_ood", "PDE rollout error vs horizon, OOD"),
        "",
        _table(pde_table(rows, "ood")),
        "",
        _fig("burgers_rollout", "Burgers rollout, conv vs dense"),
        "",
        "## Limitations",
        "",
        "- One network size per arm and a two-value learning-rate grid, tuned "
        "on one system per family; other systems use the same settings.",
        "- Training data are noise-free and sampled at one fixed dt.",
        "- Equal gradient steps is the compute rule; the unrolled loss and the "
        "RK4-based arms cost more wall time per step, reported in the tables.",
        "- The HNN arms know which coordinates are q and p; the leapfrog arm "
        "also assumes a separable Hamiltonian. Neither enforces rotation "
        "invariance, so angular momentum on Kepler is not protected.",
        "- The PDE `physics` reference uses the right equation; it shows what "
        "the equation buys on this grid and is not a competitor.",
        "",
        "## References",
        "",
        "- S. Greydanus, M. Dzamba, J. Yosinski, Hamiltonian Neural Networks, "
        "NeurIPS 32 (2019), arXiv:1906.01563.",
        "- Z. Chen, J. Zhang, M. Arjovsky, L. Bottou, Symplectic Recurrent "
        "Neural Networks, ICLR (2020), arXiv:1909.13334.",
        "- Y. Yin et al., Augmenting physical models with deep networks for "
        "complex dynamics forecasting (APHYNITY), J. Stat. Mech. (2021) 124012; "
        "`docs/references.bib: yin2021aphynity`.",
        "- R. T. Q. Chen, Y. Rubanova, J. Bettencourt, D. Duvenaud, Neural "
        "Ordinary Differential Equations, NeurIPS 31 (2018), arXiv:1806.07366.",
        "- J. Pathak, B. Hunt, M. Girvan, Z. Lu, E. Ott, Model-free prediction "
        "of large spatiotemporally chaotic systems from data, Phys. Rev. Lett. "
        "120, 024102 (2018).",
        "- E. N. Lorenz, Deterministic nonperiodic flow, J. Atmos. Sci. 20, 130 "
        "(1963).",
        "- J. C. Sprott, Chaos and Time-Series Analysis, Oxford University "
        "Press (2003).",
        "- G. Benettin, L. Galgani, A. Giorgilli, J.-M. Strelcyn, Lyapunov "
        "characteristic exponents for smooth dynamical systems, Meccanica 15, "
        "9 (1980).",
        "",
        f"Study wall time: {_fmt(meta['seconds'] / 60, 3)} min for "
        f"{meta['n_runs']} runs.",
        "",
    ]
    text = "\n".join(parts)
    path = path or get_settings().root / "docs" / DOC
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return text
