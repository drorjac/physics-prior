# Related work

Where this project sits against the work it builds on, and what it does that
the cited work does not. BibTeX for every entry is in
[`references.bib`](references.bib), each checked against its DOI or arXiv
record.

## Scope

Most physics-informed ML papers ask whether a PINN *solves* an equation, on
simulated data. This project asks what a physics prior is worth, and what
it costs when the prior is wrong, on **real measured data** (LIGO, COBE/FIRAS,
NIST, JPL) **paired with simulations** where the law is known exactly, so a
method's own error can be separated from the data's. Besides prediction error
it measures **identifiability**: whether an arm returns a physical constant
with a meaningful uncertainty. The black box it is compared
against is tuned, and negative results are reported at the same size.

## Physics-informed neural networks

**PINNs** (Raissi, Perdikaris & Karniadakis 2019, `raissi2019pinn`; review by
Karniadakis et al. 2021, `karniadakis2021piml`) put a differential equation's
residual in the loss. `relativity/gw150914` uses exactly this form (shape **A**
in the README); the other tracks use a law plus a learned correction
(shape **B**), because their law is algebraic.

**Failure modes** (Krishnapriyan et al. 2021, `krishnapriyan2021failure`)
showed PINNs can fail to train even where the equation is simple. This project
meets one such failure and reports it rather than tuning it away: the 2-D
residual PINN on the PDE rung of the neglected-terms study does not converge,
and is marked `converged=False` ([`neglected/`](neglected/)).

**Loss balancing** (Wang, Teng & Perdikaris 2021, `wang2021gradient`) is the
one PINN improvement that shipped here, chosen by ablation on the tuning seeds
against ensembles, early stopping, Fourier features and L-BFGS
([`plans/PLAN.md`](plans/PLAN.md) §5A). **Self-adaptive weights** (McClenny &
Braga-Neto 2023, `mcclenny2023sapinn`) are an alternative way to balance the
same terms; they were not tested here.

## A known model plus a learned correction

The `pinn` arm's shape B — `y = law(x; θ) + σ_y·NN(x)`, with the correction's
norm penalised — is closest to **APHYNITY** (Yin et al. 2021,
`yin2021aphynity`), which augments an incomplete physical model with a network
and keeps the augmentation minimal so the physical parameters stay
identifiable. The statistical ancestor is **model discrepancy** in the
calibration of computer models (Kennedy & O'Hagan 2001,
`kennedy2001calibration`). This project adds two dials and measures along
them: `w_phys`, how strongly the correction is suppressed, and `eps`, how
wrong the law is (hypotheses H3 and H4 in [`HYPOTHESES.md`](HYPOTHESES.md)).
At `w_phys = 0` the network absorbs the physics and the recovered constants
go wrong while the held-out error barely moves.

## Symbolic regression

The `sr` arm uses **PySR** (Cranmer 2023, `cranmer2023pysr`). **AI Feynman**
(Udrescu & Tegmark 2020, `udrescu2020aifeynman`) is the best-known
physics-inspired alternative. The finding here is about both: a symbolic
search cannot return a law outside its operator vocabulary (H6), so the
operator set is itself a prior. Planck's law is the controlled case.

## Benchmarks and baselines

**PDEBench** (Takamoto et al. 2022, `takamoto2022pdebench`) and **PINNacle**
(Hao et al. 2024, `hao2024pinnacle`) benchmark solvers on simulated PDE
fields. This project is not that kind of benchmark: its tracks are measured
data with published constants, and it scores parameter recovery and
extrapolation as well as fit.

**Weak baselines** (McGreivy & Hakim 2024, `mcgreivy2024weak`) documented how
often ML-for-PDE papers compare against under-tuned baselines. It is the
reason the `nn` arm here is grid-tuned on held-out data on the tuning seeds
before any comparison, and it is consistent with finding 1: inside the
training range, a tuned black box is competitive (H1).
