# Contributions

What this project adds, separated by how new it is and how strongly the
evidence supports it. Numbers are not repeated here; each item links to the
page where they are generated from `results/` and re-derived by
[`tests/test_claims.py`](../tests/test_claims.py). The novelty of the first
group has been checked against [RELATED_WORK.md](RELATED_WORK.md) only, not
against a full literature search.

## Likely new

**1. Gradient-norm loss balancing is unstable for law-plus-correction PINNs.**
Loss balancing (Wang, Teng & Perdikaris 2021) sets the physics weight from a
ratio of gradient norms. It was designed for PDE residuals. In a PINN of the
form `law + correction` with the penalty `mean(NN²)`, the penalty's gradient
vanishes with the correction, so the weight has no fixed point: it grows
without bound and switches the correction off, and the network becomes the
fitted law. An ablation that compares against an unbalanced network, whose
correction overfits, records that collapse as an improvement; this project's
own Phase 2 ablation did. Measured on every track, with the weight still
rising at the end of training in almost every run.
[optimization/](optimization/) §5 · [DECISIONS.md](DECISIONS.md)

**2. For a learned correction to help, the missing physics must be
distinguishable from the law and extrapolable by the network.** The project's
hypothesis H3 said distinguishability was enough. With the correction's weight
tuned rather than annealed away, the correction extrapolates worse than the
fitted law on the algebraic tracks, including helium, where the missing
quantum defect depends on inputs the law ignores. The input that leaves the
training range is `n`, and a network does not carry the defect's tail beyond
the range it saw. The helium test was named in advance and meets its own
refutation criterion. The pulsar test was pre-registered with a simulated
control: the correction learns an age dependence where one exists, so its
failure on real pulsars is a property of the data.
[HYPOTHESES.md](HYPOTHESES.md) H3 · [quantum/](quantum/) · [gravity/](gravity/)

**3. The value of a physics prior for field reconstruction grows with the
field's dimension.** From sparse sensors, a physics-constrained reconstruction
needs a number of sensors set by the physical unknowns, while a GP, RBF
interpolation and a network need close to a decade more per added dimension.
In 1-D there is no advantage; in 3-D it exceeds an order of magnitude. The
hypothesis and its refutation criteria were written before the runs, and hold
on two field families. A PDE-residual PINN is the weakest method in 3-D.
[reconstruction/](reconstruction/)

## Solid, and known in kind

**4. A good fit does not diagnose a wrong law; the fitted constant absorbs
what the law omits.** Several real-data cases, each traced to its cause: the
helium Rydberg constant from the hydrogenic law, the Newtonian chirp mass of
GW150914, the Sun's GM from a network trained without its physics term, and
Mercury's residual relativistic coefficient, which turned out to be two
omitted effects of opposite sign.
[HYPOTHESES.md](HYPOTHESES.md) H4, H5 · [relativity/](relativity/) · [quantum/](quantum/)

**5. Learned time-steppers for ODEs and PDEs, against expectations written in
advance.** Structure-preserving steppers keep invariants bounded and residual
steppers beat direct ones; neural ODEs, unrolled training losses and local
convolutional steppers did not deliver what was expected, and a coarse
classical solver beat every learned PDE stepper.
[dynamics/](dynamics/)

**6. Spatial fields on real and simulated data.** Temperature over the Alps
with the lapse rate as the prior, where kriging on the law's residuals
extrapolates best up the mountain and the ordinary least-squares error bar on
the lapse rate is too small; a simulated radio field where fitting the
path-loss law also locates the transmitter.
[fields/](fields/)

## Reproduced as checks, not claimed as findings

The QED shift of hydrogen's 1s level, helium's quantum defects, young pulsars'
braking indices below 3, the atmospheric lapse rate, Mercury's 43″/century,
and symbolic regression's dependence on its operator set. Recovering them
shows the pipeline measures what it claims to; none is new physics.

## Method

Predictions are committed to the repository before the runs that test them.
Choices are made on tuning seeds and reported on separate seeds. Every
committed number can be regenerated and compared (`make reproduce`,
[REPRODUCTION.md](REPRODUCTION.md)), every number quoted in prose is
re-derived by a test, and negative results are reported at the same size as
positive ones. This is the practice McGreivy & Hakim (2024) find missing from
much of machine learning for PDEs, and it is what caught contribution 1.

## Limits a reader should weigh

Three reporting seeds per cell; small real datasets (8 planets, 13 pulsars);
a tuning rule whose validation block sits just above the training range while
the tests run much further; a radio study that is simulation only; and
novelty checked against the related-work page, not a systematic search.
