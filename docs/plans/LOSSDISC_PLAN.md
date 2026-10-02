# Plan: learning how a PINN weighs its loss, by symbolic regression

## Question

A PINN is trained on a sum of terms: data misfit, ODE/PDE residual, and
sometimes initial or boundary conditions. How the terms are weighted, and how
the residual is weighted across the domain, decides whether training
converges. In this repository the weighting has so far been chosen by hand:
the `w_phys` dial, the warm-up and ramp found for Lorenz, the `gradnorm`
balance and the causal weights of Wang, Sankaran and Perdikaris.

The question here is whether a symbolic search can find a weighting rule that
is better than the hand-designed rules, on problems it was not searched on,
and whether the rule it finds is short enough to read.

## Object being learned

A weighting rule is an expression tree over training-state features. It is
evaluated at every training step on detached quantities, so it changes the
loss but never receives a gradient.

Trial A, per-point residual weights for forward problems:

    L = mean_i w_i r_i^2,   w_i = exp(g(t_i, C_i, q_i, tau)) / mean_j exp(g(...))

- `t` collocation time scaled to [0, 1];
- `C` cumulative mean squared residual at earlier times, sum_{j<i} r_j^2 / N
  (the input of causal weighting);
- `q` the same cumulative sum as a fraction of the total, in [0, 1];
- `tau` training progress, step / steps.

`g = 0` is the plain PINN; `g = -eps C` is causal weighting. Both are points
of the search space, so the search can only lose to them by failing to search.
The exp link keeps weights positive; the mean normalisation keeps the
effective learning rate fixed, so a rule cannot win by rescaling the loss.

Trial B, the data/physics balance for inverse problems:

    L = L_data + lambda * L_phys,   lambda = exp(h(tau, rho, gamma))

- `rho` log10(L_data / L_phys) at the current step;
- `gamma` log10 of the gradient-norm ratio |grad L_data| / |grad L_phys|,
  refreshed every 10 steps (the input of `gradnorm` balancing).

`h = const` is a fixed weight, `h = ln 10 * gamma` is gradient-norm
balancing, and `h = a + b tau` is a ramp.

## Search

Genetic programming over the repository's own expression trees
(`physprior.symbolic`), with fitness measured by training: each candidate is
used to train a PINN on every meta-training task and tuning seed, and its
fitness is the mean log10 relative error (clipped at 10). Invalid rules (NaN
or inf weights) are discarded. Parsimony is a small penalty per node.
Distinct expressions are cached, so a rule is never trained twice.

## Protocol

The repository's seed rule, with tasks split the same way as seeds:

| stage | tasks | seeds | used for |
|---|---|---|---|
| baseline tuning | meta-train | 3, 7, 19 | eps of causal, lambda of the fixed weight |
| search | meta-train | 3, 7 | GP fitness |
| selection | meta-train | 3, 7, 19 | pick one rule from the top of the search |
| report | held out | 11, 23, 42 | every number in the results table |

Held-out tasks include equations the search never saw (pendulum, van der Pol)
and frequencies outside the meta-training range. Every method gets the same
network, optimiser, learning-rate schedule, step budget and collocation
sampler; only the weighting differs. Comparisons are paired by task and seed,
with a Wilcoxon signed-rank test over the pairs.

## Failure modes planned for

- Overfitting the meta-training tasks: held-out equations, report seeds.
- Selecting on seed noise: selection on a third seed, report on three more.
- A rule that wins by scale: weights are mean-normalised in trial A.
- A rule that wins on one task and blows up on another: the error is clipped
  but kept in the mean, so a blow-up costs one decade at most and still costs.
- A weak baseline: causal eps and the fixed lambda are tuned on the same
  tasks and seeds the search uses.

## Other ways to combine SR and PINNs

Each is a candidate study for this repository. The first two are the trials
above.

1. Learned residual weights (trial A).
2. Learned data/physics balance schedules (trial B).
3. Closure discovery: a PINN with a network term for the unknown part of the
   law, then SR on that network output to write the missing term as a
   formula (universal PINN plus SR). `benchmark/neglected.py` already learns
   such a correction; SR on it is the next step.
4. Distillation: SR on a trained PINN's solution or derivatives to obtain a
   closed form, then a refit of the closed form's constants on the data.
5. SR-initialised PINN: SR on the data gives an approximate law, the PINN
   starts from it as a hard prior and learns only the residual.
6. Learned collocation density: SR for a sampling density p(t) as a function
   of the residual field, as an alternative to residual-based adaptive
   refinement.
7. Learned learning-rate or optimiser update rules, searched symbolically
   over gradient statistics.
8. Learned residual transforms: SR for phi(r) in mean phi(r_i), a robust or
   scale-adaptive residual loss.

## Deliverables

- `src/physprior/lossdisc/`: tasks, network and training, rules, search,
  study, report.
- `physprior lossdisc [--quick]`: runs both trials into `results/lossdisc/`
  and renders `docs/lossdisc/README.md`.
- `tests/test_lossdisc.py`: forward-mode derivatives against autograd, the
  baselines as special cases of the rule space, a small end-to-end search.
