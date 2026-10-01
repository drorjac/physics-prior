# Tooling: what runs, and how a formula actually comes out

Every dependency is declared in `pyproject.toml`. The heavy ones are optional
extras: the data layer, the simulations and the classical fits run without
either of them.

```bash
pip install -e '.'           # numpy, scipy, pandas, matplotlib, sympy, h5py, requests
pip install -e '.[nn]'       # + torch        -> the nn and pinn arms
pip install -e '.[sr]'       # + pysr, Julia  -> the symbolic-regression arm
pip install -e '.[all]'      # everything, including the notebook toolchain
```

Versions below match `tools/requirements.lock`, the environment the committed results were produced with.

| Package | Version | What it does here |
|---|---|---|
| **PySR** | 2.5.0 | symbolic regression; the only arm that returns a law it was not given |
| **Julia** | 1.11.9 | PySR's search engine (`SymbolicRegression.jl`), driven through `juliacall` 0.9.36; PySR resolves `SymbolicRegression.jl` 2.4.1 |
| **SymPy** | 1.14.0 | parses PySR's output into an expression tree, simplifies it, and `lambdify`s it back to a numpy function |
| **PyTorch** | 2.11.0 | the neural arms. `torch.autograd.grad` supplies the derivative in the PINN's ODE residual; the physical constants are ordinary `nn.Parameter`s |
| **SciPy** | 1.18.1 | `curve_fit` (the `physics` arm, with covariance), `solve_ivp`/DOP853 (Schwarzschild and photon geodesics), `eigh_tridiagonal` (the Schrödinger solver), `signal` (whitening, Butterworth, Hilbert, resampling) |
| **NumPy** | 2.5.3 | the integrators, the finite-difference stencils, everything else |
| **h5py** | 3.14.0 | reads the LIGO strain files |
| **requests** | 2.34.2 | all downloads (**not** `curl`; see `docs/DATA.md`) |
| **pandas / matplotlib** | 3.0.5 / 3.11.1 | results tables and figures; `PillowWriter` writes the GIFs, since ffmpeg is not installed |

---

## How a formula comes out of symbolic regression

This is the part that is usually hand-waved, so here is the whole path, with
the actual output of the gravity run at each step.

### 1 · What PySR is given

Only numbers. For the force law it is a simulated orbit's radius and
acceleration magnitude: no equation, no template, no units:

```python
fit_sr(r_scaled.reshape(-1, 1), a_scaled,
       feature_names=["r"],
       binary_operators=["+", "-", "*", "/", "^"],
       unary_operators=[],          # no exp/log: a force law is a power law
       niterations=60, maxsize=12, seed=11)
```

Two choices do real work and are stated wherever they are made:

- **The operator set is a prior.** Giving it `exp` and `log` but no `^` is why
  the FIRAS run found a Wien-like exponential instead of Planck's law. The
  operator set says what kinds of law are reachable at all.
- **Inputs and targets are scaled to O(1).** PySR searches over constants, and
  a target of order 1e-6 makes it hunt for tiny numbers instead of structure.
  A power law's *exponent* is invariant under this scaling, which is why it is
  safe; the constant is rescaled back afterwards.

### 2 · What comes back

A Pareto front (one best expression at each complexity) kept in
`fit.extra["pareto"]`. The selected one is a string:

```text
1/(r*(r/0.9386088)**0.9999969)
```

That is `r^-2`, written the way a search that multiplies and divides happens
to have found it.

### 3 · String → callable

`physprior/methods/symbolic.py::_to_fit`:

```python
expr = sympy.sympify(d["expression"])          # text -> expression tree
fn   = sympy.lambdify(syms, expr, "numpy")     # tree -> vectorised function
```

From here the discovered law is an ordinary numpy function and goes through
exactly the same scoring as every other arm.

### 4 · Callable → physics

This is the step that matters, and it is done **numerically, not by reading
the text**. `power_law_exponent` measures the slope of `log y` against
`log x` on the expression itself and checks it is constant:

```python
slopes = diff(log(fn(xs))) / diff(log(xs))
if ptp(slopes) > tol: return None      # curved in log-log: not a power law
return mean(slopes)                    # -> -1.9999969
```

Why not parse the string? Because PySR writes the same function a dozen ways
(`f*0.625*f**1.478/f**(-1.188)` is `f^3.667`), and a parser that handles one
form silently returns "no law found" for the others. Measuring the function
works on all of them.

Two bugs in this function were found by using it, both now in the tests:

- it required `y > 0`, so it reported "not a power law" for **every bound-state
  spectrum**, whose energies are negative. It now takes `|y|` and rejects only
  a genuine sign *change*.
- the Planck check compared a log-slope against 2 when the exact Planck
  function's slope on that interval is 1.84. **The check was wrong, not the
  answer.**

### 5 · Reading a constant off the law

Once the exponent is known, the constant is *not* taken from the log-log
intercept: that minimises error in `log a`, not in `a`, and extrapolates to
`r = 1 m`, far outside the data. With the exponent fixed, regress directly:

```python
basis = r ** -2.0
mu = basis @ a / (basis @ basis)         # -> GM, to 0.6 ppb on simulated data
```

The two estimators differ by ~60 ppm on the same numbers. That is the
estimator, not the physics, and both are reported.

### 6 · Caching and reproducibility

Every search is cached under `.cache/sr/`, keyed by a hash of the data **and**
the configuration. Deleting the cache reproduces the results because PySR runs
with `deterministic=True`, `parallelism="serial"` and an explicit
`random_state`. Sweeps use a shorter search (25 iterations) than the headline
fits; every row records which it used.

---

## How the PINN gets its derivatives

No finite differences. The network's output is differentiated with respect to
its input by autograd, at collocation points that need no data:

```python
y  = net(t_collocation)                       # t requires_grad=True
dy = torch.autograd.grad(y, t_collocation,
                         torch.ones_like(y), create_graph=True)[0]
residual = dy - fdot_pn(y, Mc)                # Mc is an nn.Parameter
loss = data_mse + w_phys * (residual ** 2).mean()
```

`create_graph=True` is what lets the residual itself be differentiated during
back-propagation, so `Mc` receives a gradient through the physics term. The
constants are optimised in log space (`value = init * exp(raw)`), which keeps
positive quantities positive and makes the search scale-free over decades.

---

## Where it runs, and the precision that follows

The default is **the CPU in float64**, and the second half of that sentence is
the reason for the first. Every physics term here differentiates the network
**twice**, and a second derivative in float32 keeps roughly half the
significant digits a first derivative keeps. The recovered constants *are* the
result, so precision is not a tuning knob.

```bash
physprior info                 # what this machine offers, and what was chosen
PHYSPRIOR_DEVICE=cuda physprior run quantum
PHYSPRIOR_DEVICE=cpu  PHYSPRIOR_DTYPE=float32 physprior run quantum   # deliberate
```

`physprior.methods.device.resolve()` returns `(device, dtype)` as a **pair**,
because on Apple Silicon they are not independent: **MPS has no float64
kernels at all**, so asking for MPS is asking for float32 whether or not you
meant it. `resolve()` warns rather than letting a second-derivative residual
quietly lose half its precision.

### Why `mps.is_available()` is False here, and it is not the hardware

This machine has an **Apple M2 with 10 GPU cores and Metal 3**. The project's
torch still cannot use it:

```
RuntimeError: The MPS backend is supported on MacOS 14.0+
```

torch 2.11 requires macOS 14; this machine runs 13.4. So `is_built()` is True
(the code is compiled in) and `is_available()` is False (the OS is too old):
an **OS constraint**, not a missing GPU and not missing code. PyTorch raised
that floor at 2.9, so `torch==2.8.0` reaches the same GPU on the same machine,
which is how the numbers below were obtained. See `tools/README.md`.

### What the GPU is actually worth

The 2-D field residual with second derivatives, ms per epoch:

| collocation points | cpu float64 | cpu float32 | mps float32 | speedup |
|---|---|---|---|---|
| 2,000 | 217.7 | 122.9 | 62.2 | 3.50× |
| 10,000 | 856.3 | 354.8 | 109.2 | 7.84× |
| 50,000 | 6455.5 | 3374.3 | 955.4 | 6.76× |

Roughly **7× once the collocation set is large**, which is the field case and
only the field case. The 1-D tracks fit in ~25 s and are bottlenecked by the
optimiser's *serial* epochs; a few hundred points do not fill a GPU.

### What float32 costs, measured

On the analytic field, where α = 0.05 exactly:

| | implied α | relative error |
|---|---|---|
| cpu float64 | 0.050000000000 | – |
| cpu float32 | 0.050000000991 | 2.0×10⁻⁸ |
| mps float32 | 0.049999999585 | 8.3×10⁻⁹ |

**Negligible, and an earlier draft of this section said the opposite.** It
argued that a float32 second derivative "loses about half its digits" and
concluded single precision was unsafe here. The digit count is right and the
conclusion was wrong: the worst pointwise error in `u_xx` is 6×10⁻⁷ relative,
while the *network's* error in `u_xx` is 60 %, eight orders of magnitude
larger. Arithmetic precision is not what limits this problem, and the warning
in `resolve()` now says so.

**The device selection inside the package is still unverified**, because the
package's own torch cannot reach the GPU on this OS. Every number in
`results/` was produced on the CPU in float64.

---

## How the simulations are integrated

| Problem | Method | Why |
|---|---|---|
| N-body orbits | velocity Verlet (`physprior/numerics/integrators.py`) | symplectic: energy error oscillates and stays bounded instead of drifting. RK4 is implemented alongside purely to show the difference: at 4-day steps over 200 years Verlet drifts 2% and RK4 drifts 412%, i.e. the orbit unbinds |
| Schwarzschild orbit, photon geodesic | `solve_ivp`, DOP853, `rtol=1e-12`, dense output | the perihelion is then located by `brentq` on `du/dφ` rather than by fitting a parabola to a sampled grid; Mercury's shift is 5e-7 rad per orbit and a grid estimate is only good to ~1e-6, i.e. bigger than the effect |
| Schrödinger, bound states | `eigh_tridiagonal` on a second-order finite-difference Laplacian | direct, and its O(dx²) error is measured by `grid_convergence` rather than assumed. Richardson extrapolation from two grids takes hydrogen from 300 ppm to 4 ppm, which is what makes the 10.8 ppm QED comparison possible at all |
| Schrödinger, time dependent | split-operator with FFTs | unitary by construction; the norm holds to 5e-15 |
| Acceleration from an ephemeris | 6th-order central differences | 4th order at a 3-hour step leaves a truncation error of a few 1e-9, which is a few per cent of the GR term and lands entirely in the fitted coefficient |

---

## Watching it happen

Training is recorded, not just reported. Passing `record_every` to any neural
arm keeps the loss split into its data and physics parts, every trainable
constant, and the model's prediction on a fixed grid:

```python
fit = fit_pinn(..., record_every=25, record_grid=grid)
fit.history["GM"]          # the constant, epoch by epoch
fit.history["phys_loss"]   # the physics term on its own
```

`physprior/viz/animate.py` turns that into a GIF with the fit on the left and
the constant walking toward its published value on the right. The same module
animates orbits and wavefunctions. All GIFs, at 100 dpi; ffmpeg is not
installed here and a GIF renders anywhere.
