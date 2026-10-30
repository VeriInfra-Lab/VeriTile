# chunk_retention_ops

Mathematics exposes weighted/suffix retention, final/initial gradient, output/query/key/local coefficient/partial-value sums, and mathOutputs/result explicitly route all eight channels.

- [chunk_retention_ops_repaired.py](../chunk_retention_ops_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Four genuine Real main entries retained: forward state, standalone output, backward state and standalone gradient. Optional history endpoints, supplied saved/future histories and partial dV scope remain explicit.

## Conditions

Original arbitrary PID forward/output domain; original StatePID premise for backward; original head and chunk hypotheses for gradient, with no new key-block bound. PublicIO masks/frames retain active-source scope. Selected Contract keeps sibling addressed/Tensor backing interfaces.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_retention_ops contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
