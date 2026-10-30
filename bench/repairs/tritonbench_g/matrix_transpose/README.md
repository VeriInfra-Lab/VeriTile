# matrix_transpose

fun xs col row => xs row col

- [matrix_transpose_repaired.py](matrix_transpose_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original single unmasked FP16 tile, m*d outputs; original positive power-of-two dimensions and <=2^20 tile contract retained.

## Conditions

Public conditions:

- InputContract m d
- Masked2DKernelIO supplies disjoint placement, input binding, actual access safety/success and full exterior frame

Contract relation:

Generated FP16 storage, original shape checks and output-coordinate convention match Contract.formula/flatFormula.

Exact FP program pairs and numerical dependencies are recorded in the [matrix_transpose contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
