# A step-by-step PINN course

Six notebooks, in order. Each one is generated from
[`physprior/reporting/tutorials.py`](../../src/physprior/reporting/tutorials.py)
and executed in CI, so nothing here is prose that has drifted from the code
it claims to teach.

```bash
physprior tutorials --execute      # build and run all six
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
| **T7** | [when the prior wins](T7_when_the_prior_wins.ipynb) | the regime the other tracks were missing: a law with a **term left out**. Where the PINN beats both `physics` and `nn` by an order of magnitude — and the condition under which it does not |

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

## What the course is honest about

A PINN is **the wrong tool** for T1's harmonic oscillator (an integrator is
microseconds and exact) and for T5's square well (a tridiagonal
diagonalisation is milliseconds and more accurate). Both are included anyway,
because having the exact answer is what makes the failure modes *measurable*.

**T7 is the one to read if you only read one.** T6 shows the `pinn` arm winning one cell in twelve on the real tracks; T7 shows why — those tracks are mostly the exact-law and the degenerate cases — and builds the third regime, where the same arm wins by an order of magnitude.

The value is in the cases the classical method cannot reach — an unknown
constant inside a differential law, a potential known only pointwise, a
geometry with no mesh — and in knowing which of the standard remedies survive
contact with real data. T6 has the measurements.
