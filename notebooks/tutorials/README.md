# A step-by-step PINN course

The notebooks, in order. Each one is generated from
[`physprior/reporting/tutorials.py`](../../src/physprior/reporting/tutorials.py)
and the topic modules next to it, so nothing here is prose that has drifted
from the code it claims to teach. T1–T7 are the PINN course; T8 onwards
widen it to optimisation, spatial fields, symbolic regression, learned
dynamics and a chaotic inverse problem.

```bash
physprior tutorials --execute      # build and run all of them
physprior tutorials T5             # just one
```

| | Notebook | What it teaches |
|---|---|---|
| **T1** | [what a PINN is](T1_what_is_a_pinn.ipynb) | a network trained on an **equation** rather than data: collocation points, autograd derivatives, why `tanh` and never `ReLU`, initial conditions as a **hard** constraint |
| **T2** | [forward and inverse](T2_forward_and_inverse.ipynb) | the two jobs. Recovering an unknown constant from sparse noisy data, and the `w_phys` dial that decides whether it means anything |
| **T3** | [gravity](T3_gravity.ipynb) | the law is **exact**. Kepler on real JPL data, five arms at matched capacity, and what a physics prior actually buys in the best case |
| **T4** | [relativity](T4_relativity.ipynb) | the law is a **truncated expansion**. A 9 M☉ bias that the RMSE cannot see, and the residual-PINN form for differential laws |
| **T5** | [quantum](T5_quantum_wavefunction.ipynb) | **no data at all**. Learning ψ and E together from an operator equation: the trivial solution, hard boundaries, orthogonality, spectral bias |
| **T6** | [when PINNs fail](T6_when_pinns_fail.ipynb) | the failure catalogue, measured on real data. Which of seven standard improvements actually help, and which make things 98× worse |
| **T7** | [when the prior wins](T7_when_the_prior_wins.ipynb) | the regime the other tracks were missing: a law with a **term left out**. Where the PINN beats both `physics` and `nn` by an order of magnitude, and the condition under which it does not |
| **T8** | [a network from scratch](T8_network_from_scratch.ipynb) | an MLP in NumPy: forward pass, hand-written backprop checked against finite differences and autograd, initialisation, and SGD through Adam written by hand |
| **T9** | [optimizers and loss functions](T9_optimizers_and_losses.ipynb) | which optimizer reaches a physics fit, a PINN and a black box, the curvature each lands in, robust losses under outliers, and what loss balancing does to the `pinn` arm |
| **T10** | [spatial fields](T10_spatial_fields.ipynb) | reconstructing a map: station temperatures and the lapse rate over the Alps, kriging, and a simulated radio field where the physics fit also finds the transmitter |
| **T11** | [how symbolic regression works](T11_how_symbolic_regression_works.ipynb) | expression trees, the size of the search space, a genetic search and its Pareto front, SINDy, and why the operator set is a prior |
| **T12** | [field reconstruction in 1-D, 2-D, 3-D](T12_field_reconstruction.ipynb) | a rod, a plate and a 3-D potential from sparse sensors; how many sensors each method needs as the dimension grows |
| **T13** | [learning the update rule](T13_learning_the_update.ipynb) | learn a pendulum's time step four ways and watch its energy; how long a learned Lorenz stepper stays valid; a Burgers rollout with a local versus a dense stepper |
| **T14** | [the butterfly and the PINN](T14_butterfly_and_the_pinn.ipynb) | a chaotic inverse problem: the butterfly effect measured, the PINN as a block diagram and its loss term by term, a vanilla PINN collapsing and the recipe that fixes it, and the PINN against a tuned black box, single and multiple shooting |

## The arc

The difficulty rises with the physics, and each domain breaks the previous
one's assumption:

```
T3  gravity      law exact, closed form, data present
T4  relativity   law is an APPROXIMATION you can truncate wrongly
T5  quantum      no data on the right-hand side at all; two unknowns at once
T7  controlled   law INCOMPLETE by a known amount -- the dial the others lack
```

By T5 the "prior" is the entire problem statement, and the failure modes that
were cosmetic in T3 are fatal.

## Limitations

A PINN is **the wrong tool** for T1's harmonic oscillator (an integrator is
microseconds and exact) and for T5's square well (a tridiagonal
diagonalisation is milliseconds and more accurate). Both are included anyway,
because having the exact answer is what makes the failure modes *measurable*.

**T7 is the one to read if you only read one.** T6 shows the `pinn` arm winning one cell in twelve on the real tracks; T7 shows why (those tracks are mostly the exact-law and the degenerate cases) and builds the third regime, where the same arm wins by an order of magnitude.

The value is in the cases the classical method cannot reach: an unknown
constant inside a differential law, a potential known only pointwise, a
geometry with no mesh, and in knowing which of the standard remedies survive
contact with real data. T6 has the measurements.
