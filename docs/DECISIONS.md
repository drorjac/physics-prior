# Decisions

Research decisions, what was weighed, and why. The planning documents in
[`plans/`](plans/) argue for options; this page records what was *chosen*.
Open items are listed at the bottom so they are not lost inside those
documents. The questions the decisions serve are in
[`HYPOTHESES.md`](HYPOTHESES.md).

## Decided

| date | decision | alternatives weighed | reason |
|---|---|---|---|
| 2026-09-25 | **Headline the data-backed claim:** a learned correction helps when the law is incomplete in a way the law cannot imitate, and costs accuracy when the law is complete (H3). | Headline the project as a PINN architecture. | This is what the evidence supports today. The architecture framing would only be defensible after new tracks land. [`plans/PLAN.md`](plans/PLAN.md) §2.1. |
| 2026-09-25 | **Next tracks: `quantum/helium`, then `gravity/pulsar_spindown`.** | `cosmology/pantheon` + `pde/burgers` (original order); Phase 6 interference tracks first. | These are the two where the law is incomplete in a known, distinguishable way, so they test H3 directly. Pantheon would be a fourth "law already right" track, and Burgers needs a protocol redesign. [`plans/PLAN.md`](plans/PLAN.md) §4. |
| 2026-09-25 | **Mercury: report what the converged residual is and is not, instead of adding a step-size error bar.** | Quote α with a statistical ± step-size systematic ([`plans/PLAN.md`](plans/PLAN.md) §1.4). | The physical question is whether the residual comes from the numerics or from the model. Richardson extrapolation shows the derivative's remaining error is negligible. Putting back DE441's own solar J2 and frame dragging moves α *away* from 1. So the residual is missing physics, with a stated, untested candidate (the barycentric n-body relativistic terms). A step-size error bar would have hidden that. [`relativity/`](relativity/). |
| 2026-09-25 | **Publish as a private GitHub repository, `drorjac/physics-prior`; release 0.2.0.** | A public repository now; the `drorjacoby` account named in the old badges. | `drorjac` is the authenticated account. The repository stays private until there is a paper, and is made public for the Zenodo DOI. |
| 2026-09 (Phase 2) | **Ship loss balancing (`balance`) as the PINN default; keep `w_phys = 1.0` as its starting value.** | ensembles, early stopping, Fourier features, L-BFGS, a re-tuned fixed `w_phys`. | Decided on the tuning seeds only. [`plans/PLAN.md`](plans/PLAN.md) §5A has the ablation. |

## Open

| item | where it is argued | what it blocks |
|---|---|---|
| The PDE-rung PINN does not converge: keep it as a documented negative result, or try a weak-form or smoothed-derivative residual. | [`neglected/`](neglected/), [`METHOD.md`](METHOD.md) | Only whether the PDE rung has a PINN result; its conclusion rests on the `physics` arm. |
| Mercury: test the barycentric EIH term as the source of the residual. | [`relativity/`](relativity/) | The explanation of H3's degenerate real-data case. |
| Phase 5 approval, its internal ordering (§2.5), and the Mie implementation (§2.4). | [`plans/PHASE5_PLAN.md`](plans/PHASE5_PLAN.md) §5 | EM, atmosphere and loss-discovery tracks. |
| Phase 6 approval and ordering against Phase 5; whether to run Q2 (a 2-D PINN, 8–20 h of compute). | [`plans/PHASE6_PLAN.md`](plans/PHASE6_PLAN.md) §7 | Interference tracks. |
| Add `-p no:faulthandler` to pytest's `addopts`. It is only needed on this machine's sandbox. | [`plans/PLAN.md`](plans/PLAN.md) §5.3 | Nothing; `PYTEST_ADDOPTS` works around it locally. |
