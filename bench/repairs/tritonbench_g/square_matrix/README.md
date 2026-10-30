# square_matrix

fun xs i => xs i * xs i

- [square_matrix_repaired.py](square_matrix_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One contiguous active masked row/window of elementwise square, not matrix multiplication and not a new complete grid/host theorem. Public IO includes safe successful source execution and frame.

## Conditions

- Original n/B and active-lane mask retained
- Public typed input/output memory binding and bounds retained
- FP has no finite-value or inactive-lane numerical requirements

Exact FP program pairs and numerical dependencies are recorded in the [square_matrix contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
