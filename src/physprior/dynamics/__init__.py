"""Learning the update rule of a dynamical system from trajectories.

A model is given pairs (u(t), u(t+dt)) sampled from a simulation with a known
law and has to learn the map F with u(t+dt) = F(u(t)). It is then iterated on
its own output for many more steps than any training trajectory contained.
The study compares steppers that differ in how much physical structure they
carry: a black-box map, a residual (Euler-like) map, a neural vector field
trained through RK4, Hamiltonian networks with and without a symplectic
integrator, known physics plus a learned closure, and, for PDEs, a local
convolutional stepper against a dense one.

Modules
-------
systems     ODE systems, reference integration and its convergence study
pdes        periodic 1-D PDEs, spectral reference solver and its convergence
steppers    the learned steppers (torch)
train       one-step and unrolled training
metrics     rollouts, error-vs-horizon, invariant drift, blow-up
study       the experiments, run(quick) -> dict
figures     figures for results/dynamics
doc         render_doc() -> docs/dynamics/README.md
"""

from __future__ import annotations

TRACK = "dynamics"
