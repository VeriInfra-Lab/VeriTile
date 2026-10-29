# rmsnorm_fused

xs row col * sqrt ((sum j:Fin width, (xs row j)^2)/width + epsilon)^(-1) * weights col.

- [rmsnorm_fused_repaired.py](rmsnorm_fused_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete full-row grid in arbitrary complete launch order, both complete source loops and runtime epsilon.

## Conditions

Explicit conditions:

- shape.Valid: positive width, width≤block≤16384 and source extent bound
- rows.Perm shape.grid
- ProgramIO exposes source precision, epsilon register, X/W spans, Y output and empty scratch; judgment includes bounds/binding, access safety, success, all outputs and frame.

Contract correspondence:

Contract keeps original FP32 CUDA/contiguity/scalar checks and featureLimit coverage check width≤block(width). Public Shape.Valid retains the coverage needed by the selected kernel; no new wrapper restriction.

Exact FP program pairs and numerical dependencies are recorded in the [rmsnorm_fused contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
