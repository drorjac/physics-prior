"""Build the notebooks -- one per physics problem, plus an overview.

`physprior notebooks [--execute] [name]`

Generated rather than hand-edited, for the same reason the numbers are: a
notebook written by hand drifts from the results it claims to show. Every
figure and number is computed from `results/` at execution time.

Animations are referenced with <img src="../figures/..."> rather than embedded,
so the notebooks stay small; they render in Jupyter opened from the project
root. Run `physprior figures` (or `make figures`) first if a GIF is
missing.
"""

from __future__ import annotations

import sys

import nbformat as nbf

from physprior.config import get_settings

HEADER = """import json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import HTML, display
from physprior.viz import plots as P
from physprior.io import load_table, load_json
from physprior.config import get_settings
from physprior.benchmark.protocol import fit_arm

RESULTS = get_settings().results_dir
P.use_style()
pd.set_option("display.width", 160, "display.max_columns", 60)

def gif(path, width=560, caption=""):
    "Show an animation from figures/ without embedding it in the notebook."
    cap = f"<div style='color:#52514e;font-size:90%'>{caption}</div>" if caption else ""
    return HTML(f"<img src='../figures/{path}' width='{width}'>{cap}")
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


# ---------------------------------------------------------------------------


def nb_overview():
    return _nb(
        [
            md("""
# physprior — what does a physics prior buy you?

A neural network can fit almost any curve. A physicist writes a law with two
constants in it. On **real measured data** — and on **simulations where the
answer is known exactly** — which should you use, and for what?

Three problems, each a subpackage under `physprior.problems`:

| Problem | Real data | Simulations |
|---|---|---|
| [`gravity`](01_gravity.ipynb) | Kepler's third law, JPL DE441 | two-body orbits, the three-body problem, the solar system |
| [`relativity`](02_relativity.ipynb) | GW150914 strain, Mercury's ephemeris | Schwarzschild orbits, inspiral waveforms, light bending |
| [`quantum`](03_quantum.ipynb) | NIST hydrogen levels, COBE/FIRAS | the Schrödinger equation, bound states and tunnelling |

Five arms everywhere: `oracle` (published law) · `physics` (law, fitted) ·
`pinn` (law + network) · `sr` (symbolic regression) · `nn` (black box).

**Why both real data and simulation.** The simulations are the control. When
the law is exactly what we put in, whatever a method fails to recover is the
*method's* error — its noise floor. Every real-data number is read against
that floor.
"""),
            code(HEADER),
            md("## Physical constants recovered, across all problems"),
            code("""
h = json.load(open(RESULTS / "headline.json"))
P.fig_parameter_recovery(h["parameter_recovery"]); plt.show()
pd.DataFrame(h["parameter_recovery"])
"""),
            md("""
## Extrapolation, across the benchmark tracks

Each arm is fitted on the *low* part of its range and asked to predict the
*high* part.
"""),
            code("""
P.fig_cross_track_extrapolation(h["extrapolation_summary"]); plt.show()
pd.DataFrame(h["extrapolation_summary"])
"""),
            md("### What each arm can do, and when the black box catches up"),
            code("""
display(pd.DataFrame(h["capability_matrix"]))
display(pd.DataFrame(h["budget_crossover"]))
pd.DataFrame(h["oracle_sanity"])
"""),
            md("""
## The findings

1. **Inside the training range, with enough clean data, the black box is
   competitive.** Physics buys little there.
2. **Outside it the gap is orders of magnitude — but the mechanism is
   identifiability, not the mere presence of a law.** Track `quantum/cmb`
   shows out-of-band error falling ~56× as the fitted band reaches into the
   Rayleigh-Jeans regime, while the *in-band* error gets worse.
   `relativity/gw150914` is the counter-control: from four faint early cycles
   every fitted arm fails, physics included.
3. **Only the physics arms return something a physicist can argue with** — and
   three times in this project the argument was worth having: QED in hydrogen,
   the post-Newtonian expansion in GW150914, and a truncation error
   masquerading as a 56σ refutation of general relativity.
4. **The network eats the physics if you let it.** At `w_phys = 0` the
   recovered `GM_sun` is ~19% wrong while the held-out error barely moves.

See [`docs/TOOLING.md`](../docs/TOOLING.md) for which package does what and
exactly how a formula comes out of symbolic regression.
"""),
        ],
        "physprior overview",
    )


def nb_gravity():
    return _nb(
        [
            md("""
# Problem: gravity

**Simulations** — the ones from a first-year mechanics course, run here so the
law can be recovered from data whose answer is known exactly.

**Real data** — Kepler's third law from the JPL DE441 ephemeris, eight planets.
"""),
            code(HEADER),
            md("""
## 1 · The integrator is part of the physics model

Velocity Verlet is symplectic: its energy error oscillates and stays bounded.
RK4 has better local accuracy and is not symplectic: its error *drifts*, and a
bound orbit slowly spirals. Mercury is used because `e = 0.206` makes
perihelion the hard part.
"""),
            code("""
ic = load_table("gravity", "integrator_comparison")
display(ic[["dt_days","steps_per_orbit","verlet_energy_drift",
            "verlet_energy_spread","rk4_energy_drift"]])
fig, ax = plt.subplots(figsize=(6.6,4.0))
ax.loglog(ic.dt_days, ic.verlet_energy_drift, "o-", color=P.ARM_COLOR["physics"],
          label="velocity Verlet (symplectic)")
ax.loglog(ic.dt_days, ic.rk4_energy_drift, "o-", color=P.ARM_COLOR["sr"], label="RK4")
P._style(ax, "Energy drift over 200 years of Mercury's orbit",
         "step size  [days]", "|dE/E| at the end of the run")
ax.legend(fontsize=9, labelcolor=P.INK_2); plt.show()
"""),
            md("## 2 · Watch them run"),
            code("""
display(gif("gravity/two_body_mercury.gif", 460,
            "Sun + Mercury, velocity Verlet. The Sun moves too: this is the "
            "centre-of-mass frame."))
"""),
            md("""
### The three-body problem

Left: the Chenciner–Montgomery **figure-eight choreography** — a genuinely
periodic solution in which all three bodies chase each other around the *same*
curve. Right: the same initial conditions, perturbed.
"""),
            code("""
display(gif("gravity/three_body_figure8.gif", 420, "the periodic choreography"))
display(gif("gravity/three_body_chaotic.gif", 420, "perturbed by 1e-3"))
"""),
            md("""
**Chaos is a measurement, not an adjective.** Two figure-eights whose initial
conditions differ by one part in 10⁹ are integrated side by side and their
separation is tracked.
"""),
            code("""
m = load_json("gravity", "problem_meta")
ly = m["simulations"]["lyapunov"]
print(f"Lyapunov exponent  lambda = {ly['lambda']:.4f} per time unit")
print(f"e-folding time            = {ly['e_folding_periods']:.2f} periods")
display(gif("gravity/lyapunov.png", 560))
"""),
            md("### And the real solar system, from real JPL initial conditions"),
            code("""
display(gif("gravity/solar_system_inner.gif", 460,
            "Only the state at the epoch comes from the ephemeris; "
            "everything after is this integrator's own Newtonian solution."))
print("energy drift over the run:", m["simulations"]["solar_system_drift"])
"""),
            md("""
## 3 · Recover the law of gravity from the simulation

Symbolic regression is given nothing but the orbit's radius and the magnitude
of its acceleration. The exponent it returns should be −2, exactly.
"""),
            code("""
fl = m["discovery"]["force_law"]
print("SR expression      :", fl["sr_expression"])
print(f"exponent, SR       : {fl['exponent_sr']:.7f}   (true -2)")
print(f"exponent, log-log  : {fl['exponent_loglog']:.7f}")
print(f"GM, direct fit     : {fl['mu_direct_rel_error_ppb']:+.1f} ppb")
print(f"GM, log-log interc.: {fl['mu_loglog_rel_error_ppm']:+.1f} ppm  <- biased estimator")
print(f"radius range       : x{fl['r_dynamic_range']:.2f}  (a circular orbit could not do this)")
"""),
            md("""
### Kepler's third law, from simulated orbits

Each planet is simulated and its period **measured from the trajectory** —
successive perihelion passages — not read from the table that went in.
"""),
            code("""
kl = m["discovery"]["kepler_law"]
display(load_table("gravity","kepler_from_simulation"))
print("SR:", kl["sr_expression"])
print(f"exponent: SR {kl['exponent_sr']:.6f} | log-log {kl['exponent_loglog']:.6f}  (true 1.5)")
"""),
            md("""
### The same law, recovered from the *chaotic* run

The trajectory is unpredictable beyond a few periods. The law behind it is
not. **Chaos is a property of the solution, not of the equation** — and
symbolic regression only ever sees the equation.
"""),
            code("""
cl = m["discovery"]["chaotic_law"]
print(f"G*m recovered : {cl['Gm_recovered']:.9f}  (true 1)  -> {cl['Gm_rel_error_ppm']:+.3f} ppm")
print(f"fit residual  : {cl['residual_rel']:.2e}   over {cl['n_samples']:,} samples")
print(f"Lyapunov      : {cl['lyapunov']:.4f}  (the trajectory really is chaotic)")
"""),
            md("""
## 4 · Watch a PINN learn

The network is fitted to a noisy radius-versus-time series from the simulated
orbit, with `GM` a trainable constant started at 55% of its true value. The
right-hand panel is the point.
"""),
            code("""
tr = m["training"]
display(gif("gravity/pinn_learning_orbit.gif", 820))
print(f"GM started at {tr['GM_initial']:.4e}")
print(f"GM ended at   {tr['GM_recovered']:.4e}   (true {tr['GM_true']:.4e})")
print(f"final error   {tr['GM_rel_error_ppm']:+.0f} ppm, from data with "
      f"{tr['noise_frac']*100:.0f}% noise")
"""),
            md("## 5 · The real data: Kepler's third law from JPL DE441"),
            code("""
from physprior.problems.gravity import kepler
prob, meta = kepler.problem()
meta = {**meta, **load_json("gravity/kepler", "meta")}
fits = {a: fit_arm(a, prob, np.arange(len(prob)), seed=11)
        for a in ["oracle","physics","pinn","sr","nn"]}
P.fig_overview(prob, fits, logx=True, logy=True,
               title="Kepler's third law: eight planets, every arm"); plt.show()
f = fits["physics"]
print(f"GM_sun fitted = {f.params['GM']:.6e} +- {f.param_sigma['GM']:.1e}")
print(f"IAU nominal   = {meta['published_GM_sun']:.6e}  "
      f"({(f.params['GM']-meta['published_GM_sun'])/meta['published_GM_sun']*1e6:+.1f} ppm)")
print("SR:", meta["headline"]["sr"]["law_check"])
"""),
            code("""
ex = load_table("gravity/kepler","extrapolation")
P.fig_extrapolation_bars(ex, "Train on the four terrestrial planets, predict the four giants")
plt.show()
ex.groupby("arm")[["nrmse_in","nrmse_out"]].median()
"""),
            md("""
**Verdict.** The simulation says the method can recover the exponent to 3×10⁻⁶
and `GM` to sub-ppb when the law is exactly right. The ephemeris then returns
`GM_sun` to tens of ppm — and that residual is *physics*, not method: the
two-body formula `P = 2π√(a³/GM)` neglects the planets' own masses and uses
the osculating rather than the mean semi-major axis.
"""),
        ],
        "Problem: gravity",
    )


def nb_relativity():
    return _nb(
        [
            md("""
# Problem: relativity

Where Newton is not enough, and by how much. Two real datasets and the
simulations that calibrate what can be claimed from them.
"""),
            code(HEADER),
            md("""
## 1 · Mercury's perihelion, by direct integration

The orbit equation for a test particle around a Schwarzschild mass is

$$\\frac{d^2u}{d\\phi^2} + u = \\frac{GM}{h^2} + \\frac{3GM}{c^2}u^2 ,\\qquad u = 1/r$$

The last term is the whole of general relativity as far as planetary orbits go.
Switch it off and the ellipse closes; switch it on and it precesses. The
simulation is run **both ways** and the precession is the difference, so the
numerical bias common to both cancels.
"""),
            code("""
m = load_json("relativity", "problem_meta")
mp = m["simulations"]["mercury_precession"]
print(f"simulated GR precession : {mp['precession_arcsec_per_century']:.4f} arcsec/century")
print(f"analytic 6 pi GM/(c^2 a(1-e^2)) : {mp['analytic_arcsec_per_century']:.4f}")
print(f"Newtonian control       : {mp['control_fraction_of_signal']:.2e} of the signal")
display(load_table("relativity","precession_vs_boost"))
"""),
            md("""
The real effect is 5×10⁻⁷ radians per orbit and completely invisible on a plot,
so the animation multiplies the GR term by 4×10⁴. The exaggeration is stated
on the figure.
"""),
            code("""
display(gif("relativity/schwarzschild_precession.gif", 460))
ld = m["simulations"]["light_deflection"]
print(f"Light bending at the solar limb:")
print(f"  general relativity, 4GM/c^2b : {ld['deflection_gr_arcsec']:.4f} arcsec")
print(f"  numerically integrated null geodesic : {ld['deflection_numeric_arcsec']:.4f} arcsec")
print(f"  Newtonian (half)             : {ld['deflection_newtonian_arcsec']:.4f} arcsec")
"""),
            md("""
## 2 · The same 43 arcsec, from the real ephemeris

Completely independent route: Mercury's acceleration is differentiated out of
JPL DE441, every planet's pull is subtracted using real positions, and what is
left is fitted for the coefficient of the 1PN Schwarzschild term. General
relativity says `alpha = 1`.
"""),
            code("""
gm = load_json("relativity/mercury", "meta")
gr = gm["gr"]
lad = gr["residual_ladder"]
display(gif("relativity/mercury/gr_residual_ladder.png", 560))
print(f"alpha = {gr['alpha_GR']:.6f} +- {gr['alpha_sigma']:.6f}   (Einstein: 1)")
print(f"implied precession = {gr['precession_arcsec_cy']:.3f} arcsec/century")
print(f"GM_sun to {gr['GM_rel_error_ppb']:+.2f} ppb")
print()
print("two independent routes to the same number:")
print(f"   simulated Schwarzschild orbit : {mp['precession_arcsec_per_century']:.3f}")
print(f"   real JPL ephemeris            : {gr['precession_arcsec_cy']:.3f}")
"""),
            md("""
### The warning this track earned

The GR term is 8×10⁻⁸ of Mercury's acceleration. At a 3-hour step with a
4th-order derivative, the *truncation error of the derivative* is a few ×10⁻⁹
and lands almost entirely in `alpha`.
"""),
            code("""
display(pd.DataFrame(gm["gr_convergence"]))
display(gif("relativity/mercury/gr_convergence.png", 600,
            "run once at 180m/order 4 and stopped, this would have been a "
            "confident 56-sigma refutation of general relativity"))
"""),
            md("""
## 3 · GW150914: the real strain

32 s of real 4096 Hz strain from both LIGO detectors, whitened, band-passed,
L1 inverted and delayed 6.9 ms, coherently summed. The instantaneous frequency
is measured **model-free** from sub-sample zero crossings — no waveform is
assumed, so fitting an inspiral law to it is not circular.
"""),
            code("""
from physprior.data.sources import gwosc as gw
tr = gw.frequency_track()
fig, axes = plt.subplots(2,1, figsize=(7.6,5.6), sharex=True)
w = (tr.strain_t>-0.16)&(tr.strain_t<0.06)
axes[0].plot(tr.strain_t[w], tr.strain_h[w], color=P.INK, lw=1.2)
axes[0].axvline(tr.t_peak_s, color=P.INK_MUTED, ls="--", lw=1.5)
P._style(axes[0], "GW150914: whitened, band-passed H1+L1 strain", None, "whitened strain")
axes[1].plot(tr.t_s, tr.f_hz, "o", color=P.ARM_COLOR["physics"], ls="-", lw=1.5)
axes[1].axvline(tr.t_peak_s, color=P.INK_MUTED, ls="--", lw=1.5)
P._style(axes[1], "model-free instantaneous frequency", "t - t$_{GPS}$  [s]", "f  [Hz]")
plt.show()
print(f"{len(tr)} usable cycles; merger (envelope peak) at {tr.t_peak_s*1e3:.1f} ms")
"""),
            md(
                "### Laid over a simulated waveform at the published chirp mass — no fitting"
            ),
            code("""
display(gif("relativity/real_vs_simulated_chirp.png", 620))
"""),
            md("""
## 4 · How much physics belongs in the loss?

Same points, same fitter, same two free parameters. The only thing that
changes is the post-Newtonian order of the law in the residual.
"""),
            code("""
gwm = load_json("relativity/gw150914", "meta")
ab = pd.DataFrame(gwm["pn_ablation"])
display(ab[["label","Mc","Mc_sigma","bias_Msun","bias_sigma","rmse_hz","converged"]])
display(gif("relativity/gw150914/pn_ablation.png", 600))
"""),
            md("""
The RMSE barely moves across these rows while the recovered mass moves by
9 M☉. **Goodness of fit does not diagnose a wrong law.**

## 5 · Is that bias real, or is it the pipeline's?

The only way to know is to inject a waveform with a **known** chirp mass into
the **real detector noise** and push it through byte-for-byte the same
extraction and fitting code.
"""),
            code("""
inj = m["discovery"]["injection"]
print(f"injected Mc = {inj['mchirp_true']:.2f} M_sun, "
      f"{inj['n_usable']}/{inj['n_trials']} usable, "
      f"SNR threshold {inj['snr_threshold']}")
for k in ("pn0","pn3"):
    if k in inj:
        d = inj[k]
        print(f"  {k}: median {d['median_Mc']:6.2f}  bias {d['bias_pct']:+6.1f}%  "
              f"scatter {d['scatter_Msun']:.2f}")
print()
print("compare with the REAL event:")
for r in gwm["pn_ablation"]:
    if r.get("converged") and r["pn_order"] in (0,3):
        print(f"  pn{r['pn_order']}: {r['Mc']:6.2f}  "
              f"bias {r['bias_Msun']/gwm['published_Mc_detector']*100:+6.1f}%")
"""),
            md("""
**The injection reproduces the real event's bias pattern.** The +25% at
Newtonian order is post-Newtonian truncation, not an artefact of the
extraction; and the 2PN answer is unbiased to about 1%.

### The threshold was calibrated on injections, never on the real event

The cycle-acceptance SNR cut was originally set to 2.0 *a priori*. Injections
show that at 2.0 the extraction admits noise-induced extra zero crossings that
read as 200 Hz where the truth is 40 Hz, and a known 31.2 M☉ comes back as 9.3.
"""),
            code("""
tc = load_table("relativity","threshold_calibration")
display(tc)
print("the value adopted is the smallest whose injection bias is under 1%")
"""),
        ],
        "Problem: relativity",
    )


def nb_quantum():
    return _nb(
        [
            md("""
# Problem: quantum

Two real spectra, and the Schrödinger equation solved directly so the laws
behind them can be recovered where the answer is known exactly.
"""),
            code(HEADER),
            md("""
## 1 · Solve the Schrödinger equation

$$-\\tfrac12 \\psi'' + V(x)\\,\\psi = E\\,\\psi$$

on a uniform grid with $\\psi = 0$ at the ends, diagonalised directly
(`scipy.linalg.eigh_tridiagonal`). Units are $\\hbar = m = 1$, so the answers
are the textbook ones.
"""),
            code("""
m = load_json("quantum", "problem_meta")
sp = m["simulations"]["spectra"]
rows = [{"system": k, "law": v["law"], "max rel error": v["max_rel_error"],
         "E_0": v["energy"][0], "E_0 exact": v["exact"][0]} for k,v in sp.items()]
display(pd.DataFrame(rows))
for k in sp:
    display(gif(f"quantum/eigenstates_{k}.png", 540))
"""),
            md("""
### Its error is measured, not assumed

Second-order finite differences: halving `dx` should quarter the error.
"""),
            code("""
display(pd.DataFrame(m["simulations"]["grid_convergence"]))
display(pd.DataFrame(m["simulations"]["box_convergence"]))
display(pd.DataFrame(m["simulations"]["r_min_sweep"]))
"""),
            md("""
The `r_min` sweep is the one worth pausing on: at `r_min = 1e-3` the hydrogen
error **stops responding to the grid entirely**. Refining `dx` does nothing,
because the limit is the inner boundary, not the step size.

## 2 · Tunnelling

Split-operator time evolution — unitary by construction, so the norm is a
machine-precision check rather than a tolerance.
"""),
            code("""
tu = m["simulations"]["tunnelling"]
display(gif("quantum/tunnelling.gif", 640))
print(f"packet energy      : {tu['energy']:.3f}")
print(f"barrier height     : {tu['barrier_height']:.3f}  "
      f"-> classically allowed? {tu['classically_allowed']}")
print(f"transmitted        : {tu['transmission']:.4f}")
print(f"norm drift         : {tu['norm_drift']:.1e}   (unitary)")
"""),
            md("""
## 3 · Recover the spectrum's law

Symbolic regression is handed nothing but `(n, E_n)` from the solver.
"""),
            code("""
display(load_table("quantum","spectrum_laws"))
for r in m["discovery"]["spectrum_laws"]:
    print(f"[{r['system']:9s}] {r['law']}")
    print(f"    SR: {r['sr_expression']}")
    if r["exponent_found"] is not None:
        print(f"    exponent {r['exponent_found']:.6f} (expect {r['exponent_expected']})")
    else:
        print(f"    not a power law -- spacing {r['level_spacing']:.6f} "
              f"+- {r['level_spacing_std']:.1e}  (the oscillator is affine in n)")
"""),
            md("""
The oscillator is the interesting one: `power_law_exponent` correctly returns
**nothing**, because `E = ω(n + ½)` is affine in `n`, not a power law. The
level *spacing* is the right statistic there, and it comes out at ω to 5×10⁻⁶.

## 4 · The real atom: NIST hydrogen levels
"""),
            code("""
from physprior.problems.quantum import hydrogen
prob, hmeta = hydrogen.problem()
hmeta = {**hmeta, **load_json("quantum/hydrogen", "meta")}
f = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
print(f"R fitted from the levels : {f.params['R']:.4f} +- {f.param_sigma['R']:.4f} cm^-1")
print(f"NIST ionisation limit    : {hmeta['ionisation_limit_icm']:.6f} cm^-1 (independent)")
print(f"Bohr (reduced-mass) R_H  : {hmeta['bohr_rydberg_H_icm']:.4f} cm^-1")
print(f"   fit vs NIST limit : {abs(f.params['R']-hmeta['ionisation_limit_icm'])/hmeta['ionisation_limit_icm']*1e9:.0f} ppb")
print(f"   limit vs Bohr     : {hmeta['limit_minus_bohr_ppm']:+.2f} ppm  <- QED + relativistic")
print("SR:", hmeta["headline"]["sr"]["law_check"])
display(gif("quantum/hydrogen/bohr_residual.png", 560))
"""),
            md("""
### The same QED, from the other direction

The simulation solves the *non-relativistic Coulomb problem* — precisely what
Bohr and Schrödinger predict. NIST measures the real atom. The difference is
not solver error and not a fitting artefact.

For this comparison to mean anything the solver has to be better than the
effect, so the levels are Richardson-extrapolated from two grids: 300 ppm → 4 ppm.
"""),
            code("""
sv = m["discovery"]["schrodinger_vs_nist"]
print(f"solver error          : {sv['solver_max_rel_error_ppm']:.1f} ppm "
      f"(plain finite differences: {sv['solver_without_richardson_ppm']:.0f} ppm)")
print(f"simulation vs NIST    : {sv['mean_gap_ppm']:+.2f} ppm")
print(f"QED by fitting Bohr   : {sv['limit_minus_bohr_ppm']:+.2f} ppm")
print()
print("two routes to the same physics, agreeing within the solver's error.")
pd.DataFrame({"n": sv["n"][1:], "gap [ppm]": np.round(sv["gap_ppm_by_n"],2)})
"""),
            md("## 5 · COBE/FIRAS, and the law symbolic regression could *not* find"),
            code("""
from physprior.problems.quantum import cmb
cprob, cmeta = cmb.problem()
cmeta = {**cmeta, **load_json("quantum/cmb", "meta")}
cfits = {a: fit_arm(a, cprob, np.arange(len(cprob)), seed=11)
         for a in ["oracle","physics","pinn","sr","nn"]}
P.fig_overview(cprob, cfits, logx=True, logy=True,
               title="COBE/FIRAS monopole: every arm"); plt.show()
from physprior.benchmark.metrics import chi2_reduced
cf = cfits["physics"]
print(f"T = {cf.params['T']:.6f} +- {cf.param_sigma['T']:.6f} K  "
      f"(Fixsen 2009: {cmeta['published_T_K']} +- {cmeta['published_T_err_K']})")
print(f"chi2/dof = {chi2_reduced(cprob.y, cf.predict(cprob.x), cprob.sigma, 1):.3f}")
print()
lc = cmeta["headline"]["sr"]["law_check"]
print("SR on FIRAS:", lc["expression"])
print(f"  deviation from Planck INSIDE the band : {lc['max_rel_dev_in_band']:.2%}")
print(f"  deviation from Planck OUTSIDE it      : {lc['max_rel_dev_outside']:.1%}")
"""),
            md("""
SR fits the FIRAS band to a fraction of a per cent and the expression it
returns is **not** the Planck function. In that band `x = hν/kT` runs from 1.2
to 11.3 — almost all Wien, almost no Rayleigh-Jeans — and there `exp(−x)` and
`1/(exp(x) − 1)` are nearly the same function. The denominator is not
identifiable from the data.

The control keeps the method, the channel count and the noise fixed and only
widens the band downwards, on a synthetic spectrum where the answer is known.
"""),
            code("""
ctrl = pd.DataFrame(json.load(open(RESULTS / "_band_control.json")))
display(ctrl[["x_min","max_rel_dev_in_band","max_rel_dev_outside","is_planck_form"]])
display(gif("quantum/cmb/band_coverage_control.png", 600))
"""),
            md("""
**In-band accuracy is anti-correlated with having found the law.** The
FIRAS-coverage run has the *best* in-band fit of the five and one of the worst
extrapolations. Fit quality on the observed range is not evidence that you
have discovered anything.
"""),
        ],
        "Problem: quantum",
    )


BUILDERS = {
    "00_overview": nb_overview,
    "01_gravity": nb_gravity,
    "02_relativity": nb_relativity,
    "03_quantum": nb_quantum,
}


def build(execute: bool = False, only: str | None = None) -> None:
    settings = get_settings()
    settings.notebooks_dir.mkdir(parents=True, exist_ok=True)
    for name, fn in BUILDERS.items():
        if only and only not in name:
            continue
        nb = fn()
        path = settings.notebooks_dir / f"{name}.ipynb"
        nbf.write(nb, path)
        print(f"wrote {path.relative_to(settings.root)}")
        if execute:
            from nbclient import NotebookClient

            print("  executing ...", flush=True)
            client = NotebookClient(
                nb,
                timeout=3600,
                kernel_name="physprior",
                resources={"metadata": {"path": str(settings.root)}},
                allow_errors=True,
            )
            client.execute()
            nbf.write(nb, path)
            errs = [
                c
                for c in nb.cells
                if any(o.get("output_type") == "error" for o in c.get("outputs", []))
            ]
            print(f"  done, {len(errs)} cell(s) with errors")


if __name__ == "__main__":
    build(
        execute="--execute" in sys.argv,
        only=next((a for a in sys.argv[1:] if not a.startswith("-")), None),
    )
