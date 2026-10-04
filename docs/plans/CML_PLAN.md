# Plan: rain from microwave links, physics law against network against hybrid

## Question

Rain attenuates a microwave link by A = k R^alpha L (ITU-R P.838-3). A
network can learn the attenuation-to-rain map from gauge labels instead. The
hybrid of Jacoby et al. (ICASSP 2026) runs both in parallel and fuses them
with a learned gate,

    r_hat = g r_M + (1 - g) r_D,   g = sigmoid(w . s + b),
    s = [sd(x), mean(x), log(1 + r_M), log(1 + r_D), r_M - r_D].

The question is when each wins, measured on every CML archive on this
machine and on simulations where the truth is known. The numbers of the
paper are not reused: the setup, splits and results here are built from
scratch.

## Arms

| arm | knows the law | trained parameters |
|---|---|---|
| `pl_itu` | k, alpha from ITU-R P.838-3 | none |
| `pl_cal` | the same law | k, alpha per link |
| `gru` | no | 2-layer GRU, 64 units |
| `hybrid_joint` | yes | PL + GRU + gate, end to end |
| `hybrid_gate` | yes | gate only, branches pretrained and frozen |
| `hybrid_phased` | yes | gate, then gate + GRU, then all |

All arms see the same input: excess attenuation x = A - baseline, the
baseline a causal 24-h rolling median per link. The power-law branch uses
the minutes of the target bin; the GRU and the gate see the preceding
history window.

## Data

| dataset | where | signal | reference |
|---|---|---|---|
| OpenMRG | Gothenburg, JJA 2015 | TSL - RSL, 10 s to 1 min, 28-40 GHz | city gauges within 3 km, 1 min |
| OpenRainER | Emilia-Romagna, 2021-22 | TSL - RSL, 1 min, ~25 GHz | gauge-adjusted radar or gauges |
| Netherlands | 2012 | min/max RSL, 15 min | KNMI hourly gauges within 3 km |
| OpenMesh | New York, 2023-26 | -RSL, 1 dB, 5-60 GHz | PWS / ASOS within 3 km |
| simulation | 20 links with real f, L | ITU law + noise + quantisation (+ wet antenna) | the simulated rain |

Each archive is reduced by `physprior.cml.sources.<name>.build()` to one
format (`sources/common.py`). The data stay under `~/data/cml`.

Simulation: hourly wet/dry Markov chain with a 7 % wet fraction, rain inside
wet hours either i.i.d. Gamma or AR(1)-Gamma, Gamma and AR parameters fitted
to the OpenMRG gauge record, Gaussian noise of the dry-period sd, 0.3 dB
quantisation, and optionally wet-antenna attenuation with exponential rise
and decay up to 2.3 dB.

## Protocol

- Split by calendar day, 70/15/15 train/validation/test, days stratified by
  daily rain, the same split for every arm. The split never depends on a
  model seed.
- Model seeds: tuning 3/7/19, reporting 11/23/42.
- Early stopping on the validation days only.
- Metrics on every test bin: NRMSE = RMSE / mean(r), NBIAS = mean(r_hat - r)
  / mean(r), correlation, and the same on wet bins.
- Data efficiency: train on 5, 15, 40 and 100 % of the training days.
- Gate behaviour: mean gate against noise sd on the simulations.

## Expected failure modes

- The reference is a point gauge, the link a path: a floor on every arm.
- Wet-antenna attenuation and baseline errors bias the law upward at low
  rain.
- A network with many training days may beat the law outright; the hybrid
  is then judged against the better of its two branches.

## After this

The gated hybrid is a general arm. Once it works here it is added to the
other `physprior` tracks as `hybrid`: physics-law branch, network branch,
learned gate, against `physics`, `nn` and `pinn` on the same splits.

## Changes made while running, and why

Each change below was made after looking at a result, so each is listed
with the result that prompted it. All of them apply to every arm that uses
the affected piece; none was tuned on test days.

1. The power law gets a per-link dead zone tau (three robust noise sd,
   label-free). Without it, quantisation flicker on dry minutes read as
   rain: on OpenMRG the bare ITU law had NRMSE 51 and NBIAS +8.9.
2. Links whose label-free noise sd exceeds max(0.6 dB, 1.5 quantisation
   steps) are dropped before modelling. On OpenMRG one link with sd 1.2 dB
   carried most of the error. The limit was first a flat 0.6 dB, which on
   1-dB data dropped quiet links (9 of 40 on OpenRainER, 11 of 37 in the
   Netherlands, 8 of 10 on OpenMesh).
3. Training draws wet and dry bins 1:1 for speed, and weights them back to
   the natural mix in the loss. Without the weights the network arms were
   biased upward by +70 to +100 % (they learned that rain is common).
4. The dead zone is never below one quantisation step. On 1-dB data a
   single flicker otherwise reads as rain.
5. The gate has a learned bias per link. The paper fits one model per
   link, so each link has its own gate; one shared gate could not switch
   off the power law on OpenMesh's 5 GHz links, where it is useless (the
   phased hybrid scored NRMSE 14.8 against the GRU's 5.2).
6. Training budget: at most 20 000 wet and 20 000 dry bins per epoch, 15
   epochs, patience 3, history 30 min, because the machine was shared and
   the full budget would have taken about 20 hours.
7. For the gate-only and phased hybrids, each link's gate starts at that
   link's least-squares blend of the two frozen branches on its training
   bins. A zero start left the 5 GHz OpenMesh links at g = 0.5 for most of
   training.
8. The per-link part of the gate covers its weights as well as its bias.
   With a shared slope the gate learned on good links that a large power-law
   estimate means heavy rain, and applied that rule on a broken link.
9. OpenMesh is reported as it is. Its hybrid error is dominated by one link
   on one test day (`158_sublink_1`, 5.8 GHz, 0.58 km, 18 December 2023):
   up to 17.7 dB of excess attenuation, impossible from rain at that
   frequency and length, which the power law reads as up to 104 mm/h. No
   training or validation day contains anything like it. The day is not
   removed after the fact; the results add the median over links of the
   per-link NRMSE, which one bad link cannot dominate.
10. Early stopping scores the starting state before the first epoch, so a
    stage that only makes the validation loss worse returns the model it
    was given. Before this fix, with 5 % of the training days, calibration
    made the power law worse than the ITU law it started from, and the
    hybrids built on it inherited that.
11. OpenMesh keeps only its 24 and 60 GHz sublinks (owner's decision). At
    5 GHz the attenuation of most rain is below the 1 dB quantisation step.
