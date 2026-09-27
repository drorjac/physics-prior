# Learning the update rule: expectations, written before the study was run

This page was written before any reported-seed run of the dynamics study. It
states what each comparison was expected to show and, for each, the measured
outcome that would count against it. The criteria are implemented in
`physprior.dynamics.study.EXPECTATIONS`, which evaluates them on the
reporting seeds (11, 23, 42) and writes `results/dynamics/expectations.json`.
`docs/dynamics/README.md` shows the verdicts; nothing here is edited after
the fact.

Definitions used below. "Rollout" means iterating the learned map on its own
output from a true initial state. "Valid horizon" is the number of steps
before the rollout error first exceeds 0.1 training standard deviations
(0.4 for Lorenz-63, following Pathak et al. 2018). "Majority" means more
than half of the (system, arm) comparisons named in the criterion, each
compared on the median over the three reporting seeds.

The unroll length was reduced from 8 to 5 steps for compute before any
reported run; the criteria are unchanged.

## E1. Symplectic structure bounds energy error

Expectation: the separable Hamiltonian network with a leapfrog update keeps
its energy error bounded over rollouts ten times longer than training, while
the black-box direct, residual and neural-ODE steppers drift.

Refuted if, on the pendulum or on Kepler (both, in-distribution initial
states), the leapfrog stepper's energy error in the last tenth of the rollout
is not lower than that of every black-box stepper, or if its late/early
energy-error ratio is above 10 (not bounded).

## E2. A Hamiltonian field without a symplectic integrator still drifts

Expectation: the Greydanus-style HNN integrated with RK4 has lower energy
error than the black boxes but grows (late/early ratio above that of the
leapfrog stepper), so the integrator carries part of the benefit.

Refuted if the RK4 HNN's late/early energy-error ratio is not larger than the
leapfrog stepper's on both conservative systems.

## E3. Residual steppers beat direct steppers

Expectation: u + dt NN(u) has a longer valid horizon than u' = NN(u), both
in-distribution and from unseen initial states, because the direct map has
to learn the identity to the accuracy of one step.

Refuted if the direct stepper's valid horizon is at least the residual
stepper's on a majority of the four ODE systems (in-distribution), or on a
majority of the three non-chaotic systems out of distribution.

## E4. Training through RK4 beats an Euler-like residual

Expectation: the neural ODE (vector field integrated with RK4 during
training) has a longer valid horizon than the residual stepper.

Refuted if the neural ODE's valid horizon is not longer than the residual
stepper's on a majority of the four ODE systems.

## E5. Known physics plus a learned closure beats both black boxes when the physics is incomplete

Expectation: on the systems where a closure is defined (pendulum given the
small-angle law, Duffing given the undamped linear oscillator, Lorenz given
its linear part, Burgers given the viscous term), the closure stepper has the
longest valid horizon of the learned black boxes, with the largest margin on
unseen initial states.

Refuted if, on a majority of the closure systems, the closure stepper's
out-of-distribution rollout error at the training horizon is not lower than
that of every black-box stepper (for Lorenz, which has no out-of-distribution
set, the in-distribution valid time is used).

## E6. An unrolled loss stabilises long rollouts

Expectation: training on a 5-step unrolled loss instead of the one-step
loss lengthens the valid horizon of the same architecture.

Refuted if the unrolled loss does not lengthen the valid horizon on a
majority of the (system, arm) pairs in the objective experiment.

## E7. Locality and translation equivariance help on PDEs

Expectation: the kernel-5 convolutional stepper beats the dense MLP stepper
on long rollouts of heat, advection and Burgers, and the gap is largest with
the fewest training trajectories and on unseen (higher-wavenumber) initial
conditions.

Refuted if the dense stepper's error at the end of the rollout is at most
the convolutional stepper's on a majority of the three PDEs, or if the
conv/dense error ratio on Burgers is not smaller at 2 trajectories than at
32.

## E8. In chaos, one-step accuracy decides the valid time

Expectation: on Lorenz-63 every learned stepper stays predictable for a few
Lyapunov times at most, and the ordering of valid times follows the
ordering of one-step errors, whatever the structure.

Refuted if the Spearman rank correlation between one-step error and valid
time across Lorenz arms and seeds is above -0.5.

## E9. Structure matters most with little data

Expectation: with 2 training trajectories the structured pendulum steppers
(leapfrog HNN, closure) beat the residual black box by more than they do
with 32.

Refuted if the ratio of residual to leapfrog valid horizon is not larger at
2 trajectories than at 32.
