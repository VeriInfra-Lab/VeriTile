# softmax_triton2

fun xs i => exp(xs i) / sum j : Fin n, exp(active cover xs j)

- [softmax_triton2_repaired.py](softmax_triton2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One B-lane masked row, not an entire row-grid pipeline. The logical denominator contains exactly n terms.

## Conditions

- 0 < B
- n <= B in Correct headline
- physical backing, active input binding and exterior frame in MaskedKernelIO judgment

Exact FP program pairs and numerical dependencies are recorded in the [softmax_triton2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
