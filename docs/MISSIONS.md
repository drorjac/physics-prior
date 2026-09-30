# Missions

Every task in the project, with its goal and what it concluded about the
PINN and the physics prior. Missions with a clear, meaningful result come
first; missions whose result is weak or unresolved are listed after them as
in progress, with what would move them. Numbers live on the linked pages,
where they are generated from `results/` and re-derived by tests.

A one-notebook tour of the results is [`notebooks/summary.ipynb`](../notebooks/summary.ipynb).

## Results

| Mission | Goal | Conclusion |
|---|---|---|
| [Lorenz-63](lorenz/) | Reconstruct a chaotic trajectory from 40 noisy points, recover its three constants from a start wrong by a factor of two, and forecast | The PINN beats a tuned black box by about an order of magnitude on the trajectory and its derivative, on every reporting seed, and returns the constants. Multiple shooting matches it on the constants at 40 points; at 10 points only the PINN recovers them. Single shooting fails on most seeds, and so does the vanilla PINN; a data warm-up with a ramped physics weight is what fixes it. |
| [GW150914](relativity/) | Extrapolate a black-hole inspiral's chirp from its early cycles | The ODE-residual PINN extrapolates best of the fitted arms on every reporting seed. The law is a truncated expansion, and the network carries what the truncation drops. The Newtonian law gives a confidently wrong chirp mass that the fit quality does not reveal. |
| [When a prior helps](neglected/) | Dial how wrong the law is, in a controlled study, and find where the correction pays | A learned correction beats both the fitted law and the black box when the missing term is distinguishable from the law; when it has the law's own shape the fitted constant absorbs it instead. |
| [Pulsars](gravity/) | Test the vacuum-dipole spin-down law on real pulsars | The law is incomplete: young pulsars brake below n = 3. In a simulated control the PINN learns an age dependence; on real pulsars there is none to learn, and the tuning rule turns the correction off. All four predictions, committed before the run, hold. |
| [Field reconstruction, 1-D to 3-D](reconstruction/) | How many sensors each method needs to map a field, as the dimension grows | The physics prior's advantage grows with dimension, from none in 1-D to over an order of magnitude in 3-D. The winner is the physics-constrained fit; the PDE-residual PINN is not. |
| [Learned dynamics](dynamics/) | Learn the update rule of ODEs and PDEs and roll it forward | A Hamiltonian network with a symplectic step keeps energy bounded where black boxes drift, and residual steppers outlast direct ones. |
| [Weather over the Alps](fields/) | Map temperature from station data and predict the mountain tops | The lapse-rate law plus kriging on its residuals extrapolates up the mountain best; a black box trained on the valleys cannot. The lapse rate is recovered, and its least-squares error bar is shown to be too small. |
| [Constants from real data](RESULTS.md) | Recover physical constants with an error bar | Hydrogen's QED shift, Mercury's relativistic precession once the missing solar terms are restored, the Sun's GM and the CMB temperature, each by the `physics` arm. |
| [Optimization](optimization/) | What training choices do to physics fits, PINNs and black boxes | Second-order methods reach a physics fit in a handful of steps. Loss balancing, as used before, collapses a law-plus-correction PINN onto the plain law; it has been removed. |
| [Symbolic regression](theory/symbolic_regression.md) | How a formula search works and where it stops | The operator set is a prior: a law outside it is not found, however much data there is. |
| [Physics weight vs data](HYPOTHESES.md) (H7) | Tune the PINN's physics weight to the amount of training data | Supported, narrowly: the weight chosen for few points is at least the one for many on 4 of 5 tracks, and the per-budget weight is no worse than a single weight at 17 of 25 cells. It helps a little on the weather track; the choice is noisy. |

## In progress

| Mission | Where it stands | What would move it |
|---|---|---|
| [Helium](quantum/) | With a tuned weight, the correction extrapolates worse than the hydrogenic law; H3 is refuted there as stated. | A correction that extrapolates in n, e.g. one that learns a quantum defect rather than an energy offset. |
| [Algebraic tracks](RESULTS.md) (Kepler, hydrogen, CMB) | Where the law is already right, the tuned correction extrapolates worse than the law. | The per-budget weight study (H7), and a validation rule that reaches as far out as the test. |
| [Radio field](fields/) | The law wins in free space; with walls, GP-based methods do better. A Helmholtz-residual PINN is under-determined from signal strength alone. | Phase or boundary information for the residual PINN. |
| [3-D PDE-residual PINN](reconstruction/) | The weakest method in 3-D at the budgets run. | Longer training and a tuned PDE weight. |
| [PDE rung of the controlled study](neglected/) | The residual PINN does not converge; the rung's conclusion rests on the `physics` arm. | A weak-form or smoothed-derivative residual. |
