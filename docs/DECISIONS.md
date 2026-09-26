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
| 2026-09-25 | **Mercury: explain the converged residual physically, instead of adding a step-size error bar.** | Quote α with a statistical ± step-size systematic ([`plans/PLAN.md`](plans/PLAN.md) §1.4). | The physical question is whether the residual comes from the numerics or from the model. Richardson extrapolation shows the derivative's remaining error is negligible. Putting back what DE441 has and the model lacks (the Sun's J2 and frame dragging, and the barycentric n-body EIH terms) takes α to 1 within its error. The two omissions had nearly cancelled. A step-size error bar would have hidden all of this. [`relativity/`](relativity/). |
| 2026-09-25 | **Keep the one-body Mercury model as the headline fit; report the complete model beside it.** | Make DE441's full model the reported α. | The simple model is the one whose failure teaches something, and its generated table row is unchanged. The complete model's α is a consistency check, since DE441 assumes GR, not a better measurement. |
| 2026-09-25 | **Publish as a private GitHub repository, `drorjac/physics-prior`; release 0.2.0.** | A public repository now; the `drorjacoby` account named in the old badges. | `drorjac` is the authenticated account. The repository stays private until there is a paper, and is made public for the Zenodo DOI. |
| 2026-09-26 | **Do not add `-p no:faulthandler` to pytest's `addopts`.** | Add it to `pyproject.toml` so the suite runs unmodified on the development machine. | The `PermissionError` comes from this machine's sandbox, not from the package. CI and other machines do not need the flag. Hardcoding it would disable faulthandler's crash tracebacks for everyone. Locally, pass it on the command line or set `PYTEST_ADDOPTS`. [`plans/PLAN.md`](plans/PLAN.md) §5.3. |
| 2026-09 (Phase 2) | **Ship loss balancing (`balance`) as the PINN default; keep `w_phys = 1.0` as its starting value.** | ensembles, early stopping, Fourier features, L-BFGS, a re-tuned fixed `w_phys`. | Decided on the tuning seeds only. [`plans/PLAN.md`](plans/PLAN.md) §5A has the ablation. |

## Open

| item | where it is argued | what it blocks |
|---|---|---|
| The PDE-rung PINN does not converge: keep it as a documented negative result, or try a weak-form or smoothed-derivative residual. | [`neglected/`](neglected/), [`METHOD.md`](METHOD.md) | Only whether the PDE rung has a PINN result; its conclusion rests on the `physics` arm. |
| Phase 5 approval, its internal ordering (§2.5), and the Mie implementation (§2.4). | [`plans/PHASE5_PLAN.md`](plans/PHASE5_PLAN.md) §5 | EM, atmosphere and loss-discovery tracks. |
| Phase 6 approval and ordering against Phase 5; whether to run Q2 (a 2-D PINN, 8–20 h of compute). | [`plans/PHASE6_PLAN.md`](plans/PHASE6_PLAN.md) §7 | Interference tracks. |
