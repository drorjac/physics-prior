"""T10: spatial fields. Generated as an executable notebook.

The weather section is built here by `weather_cells()`. Further sections on
spatial fields are appended when their module provides `rf_cells()` in
`physprior.problems.fields.rf_tutorial`; it is imported lazily so this
notebook builds with or without it.
"""

from __future__ import annotations

import importlib

from physprior.reporting.cells import SETUP, _nb, code, md


def weather_cells() -> list:
    return [
        md("""
# T10 · Spatial fields: temperature over the Alps

The earlier notebooks fit laws of one variable. Here the target is a field:
monthly mean air temperature at about two hundred weather stations between
Lyon and Salzburg, from sea level to the Jungfraujoch at 3576 m.

The physics prior is the lapse rate. Temperature falls with height, and to
first order linearly:

$$T = T_0 + a\\,(\\mathrm{lon}-\\mathrm{lon}_0) + b\\,(\\mathrm{lat}-\\mathrm{lat}_0) + \\Gamma z .$$

The reference value of $\\Gamma$ is the ICAO / ISO 2533 standard atmosphere,
$-6.5$ K/km. That is a free-atmosphere number. Near the ground, along a
mountain slope, the lapse rate is set by the boundary layer and varies with
season and time of day, so the fitted $\\Gamma$ is compared with $-6.5$ as a
reference rather than as a truth.

The question is the one this project asks everywhere: what does the law buy
when a model has to predict where it has seen no data? Here that means the
mountain tops.
"""),
        code(
            SETUP
            + """
from physprior.benchmark.protocol import fit_arm
from physprior.problems.fields import weather as W
from physprior.problems.fields import weather_sim as S
"""
        ),
        md("""
## 1 · The data

NOAA's Integrated Surface Database publishes hourly observations for
thousands of stations as ISD-Lite files: fixed-width text, temperature in
tenths of a degree, missing values as -9999. The loader downloads each
station-year once, checks the column layout and the units, and forms one
number per station by a rule fixed before any fit: the mean of the 12 UTC
temperatures over the month, for stations with a reading on at least 80% of
the days.
"""),
        code("""
fld = W.load_field("july")
print(fld.rule.describe())
print(len(fld), "stations")
fld.frame().sort_values("elev_m").tail(8)
"""),
        code("""
prob, meta = W.problem("july")
z, lon, lat = prob.x.T
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
sc = ax[0].scatter(lon, lat, c=z * 1000, cmap="viridis", s=20)
fig.colorbar(sc, ax=ax[0], label="height [m]")
ax[0].set_xlabel("longitude"); ax[0].set_ylabel("latitude")
ax[1].scatter(z, prob.y, s=12, color=P.INK_2)
ax[1].set_xlabel("height z [km]"); ax[1].set_ylabel("T [degC]")
ax[1].set_title("July 2023, 12 UTC monthly mean")
plt.show()
"""),
        md("""
## 2 · Fit the law

The law is linear in its four constants, so the `physics` arm is ordinary
least squares, with an error bar from the covariance. That error bar assumes
the residuals are independent. They are not: stations in the same valley
share weather. Section 5 measures how much that matters.
"""),
        code("""
full = fit_arm("physics", prob, np.arange(len(prob)), seed=11)
for k in ("T0", "a", "b", "Gamma"):
    print(f"{k:6s} = {full.params[k]:8.3f} +- {full.param_sigma[k]:.3f}")
print(f"ICAO standard atmosphere: Gamma = {W.GAMMA_STD}")
res = prob.y - full.predict(prob.x)
print(f"residual RMS {res.std():.2f} K")
"""),
        md("""
## 3 · Extrapolate up the mountain

Train on the lowest 75% of stations by height, predict the highest 25%. The
black-box network has never seen a station above the training ceiling; the
law has a term that says what happens there.
"""),
        code("""
itr, ite = W.elevation_split(prob)
print(f"train z <= {prob.x[itr, 0].max():.2f} km, test z up to {prob.x[ite, 0].max():.2f} km")
fits = {arm: fit_arm(arm, prob, itr, seed=11) for arm in ("physics", "pinn", "nn")}
for name, f in fits.items():
    e = f.predict(prob.x[ite]) - prob.y[ite]
    print(f"{name:8s} test RMSE {np.sqrt(np.mean(e**2)):5.2f} K   "
          f"mean dT/dz {W.effective_lapse_rate(f, prob.x):6.2f} K/km")
"""),
        code("""
fig, ax = plt.subplots(figsize=(7, 4.4))
ax.scatter(prob.x[itr, 0], prob.y[itr], s=10, color=P.GRID, label="train")
ax.scatter(prob.x[ite, 0], prob.y[ite], s=14, color=P.INK, label="test")
for name, f in fits.items():
    ax.scatter(prob.x[ite, 0], f.predict(prob.x[ite]), s=14,
               color=P.ARM_COLOR[name], label=name)
ax.axvline(prob.x[itr, 0].max(), ls=":", color=P.INK_MUTED)
ax.set_xlabel("height z [km]"); ax.set_ylabel("T [degC]"); ax.legend()
plt.show()
"""),
        md("""
The network's mean `dT/dz` is the derivative of what it learned, taken by
central differences at every station. It is a way to ask a black box what it
believes about height without reading its weights.
"""),
        md("""
## 4 · Kriging

Meteorologists interpolate station data with Gaussian processes (kriging).
Two versions, both with hyperparameters chosen by maximum marginal
likelihood on the training stations:

* **ordinary kriging**: a constant mean plus a GP over east, north and
  height. A black box with a smoothness prior.
* **universal kriging**: the lapse-rate law as the mean, plus a GP on what
  the law leaves. This is the standard method for mapping temperature in
  mountains, and the fairest competitor the law-based arms have.
"""),
        code("""
for kind in ("gp", "uk"):
    f = W.fit_kriging(prob, itr, kind)
    e = f.predict(prob.x[ite]) - prob.y[ite]
    hp = {k: round(v, 3) for k, v in f.extra["hyper"].items()}
    print(f"{kind}: test RMSE {np.sqrt(np.mean(e**2)):.2f} K  hyper {hp}  "
          f"converged {f.extra['converged']}")
    if kind == "uk":
        print(f"    Gamma = {f.params['Gamma']:.2f} +- {f.param_sigma['Gamma']:.2f} K/km")
"""),
        md("""
A length scale that ends at its bound has not converged, whatever the
optimiser reports, and is flagged. For ordinary kriging the horizontal
length scale tends to run to its upper bound: the likelihood wants a
smooth regional trend, which a constant-mean GP can only imitate with a
very long correlation length.
"""),
        md("""
## 5 · The inversion case

In January the valleys fill with cold air. Over the lowest few hundred
metres temperature can rise with height, and a single linear lapse rate is
the wrong law. This is a known failure, and it is what the second case is
for.
"""),
        code("""
pj, _ = W.problem("january")
fj = fit_arm("physics", pj, np.arange(len(pj)), seed=11)
print(f"January Gamma = {fj.params['Gamma']:.2f} +- {fj.param_sigma['Gamma']:.2f} K/km")
rj = pj.y - fj.predict(pj.x)
fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(pj.x[:, 0], rj, s=12, color=P.INK_2)
ax.axhline(0, color=P.INK_MUTED, lw=1)
ax.set_xlabel("height z [km]"); ax.set_ylabel("data - law [K]")
ax.set_title("January: residual of the linear law against height")
plt.show()
"""),
        md("""
## 6 · A simulated control

On real data a wrong $\\Gamma$ could be the method's fault or the law's.
The simulation separates them: the same station positions, a field with a
known lapse rate, a spatially correlated residual drawn from a GP, and
optionally a cold pool. In the `law` case the least-squares $\\Gamma$ should
be right on average; in the `inversion` case the best linear $\\Gamma$ is
shallower than the free-air one, by an amount that is known.
"""),
        code("""
xs = prob.x
for case in S.CASES:
    g = []
    for seed in range(1000, 1020):
        p, _ = S.problem(xs, case, seed)
        g.append(fit_arm("physics", p, np.arange(len(p)), seed).params["Gamma"])
    print(f"{case:9s} truth {S.TRUTH['Gamma']:.2f}, best linear "
          f"{S.best_linear_gamma(xs, case):.2f}, physics mean {np.mean(g):.2f} "
          f"sd {np.std(g):.2f}")
"""),
        md("""
## 7 · The full study

`physprior run fields` runs every arm (including PINN and symbolic
regression) on three reporting seeds, on the elevation split and on four
longitude blocks, plus the kriging baselines, the budget and noise sweeps,
the simulated control and an error-bar coverage study. The tables are in
`results/fields/weather/` and the write-up is `docs/fields/weather.md`.
"""),
        code("""
from physprior.io import load_table
try:
    ex = load_table(W.track("july"), "extrapolation")
    kr = load_table(W.track("july"), "kriging")
    print(ex.groupby("arm")[["nrmse_out", "dTdz"]].median().round(3))
    print(kr[kr.sweep == "elevation"][["arm", "nrmse_out", "dTdz"]].round(3))
except FileNotFoundError:
    print("run `physprior run fields` first")
"""),
    ]


def _extra_cells() -> list:
    try:
        mod = importlib.import_module("physprior.problems.fields.rf_tutorial")
    except ImportError:
        return []
    return list(mod.rf_cells())


def t10_spatial_fields():
    return _nb(weather_cells() + _extra_cells(), "T10 - spatial fields")
