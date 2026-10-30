# matrix_reduction

fun xs row _ => sup′ (Fin N) (fun col => xs row col)

- [matrix_reduction.py](matrix_reduction.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Original whole-row one-block source; every row/output, M=0 allowed; N>0 is explicit. Both FP16 and FP32 storage profiles.

## Conditions

Public conditions:

- 0 < N
- MaskedTileKernelIO input and output binding, bounded accesses and exterior frame

Contract relation:

The wrapper generates contiguous typed storage (InputStorage/InputValues), rather than accepting caller tensors. Original generated storage and nonempty-row mathematical scope retained.

Exact FP program pairs and numerical dependencies are recorded in the [matrix_reduction contract review](../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
