# vector_addition_custom

Direct logical pointwise Real arithmetic.

- [vector_addition_custom_repaired.py](vector_addition_custom_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One canonical actual FP32 CTA, block size 16, mask pid*16+i<n; regional read-only aliases and exact frame retained.

## Conditions

- Regional masked IO supplies actual binding/active address safety and output frame
- Contract explicit contiguous CUDA FP32 same-device/same-shape/rank-1, extent n<2^31, no negative views, nonempty alignment
- FP actual additionSite retains active finite loaded-operand domain; inactive lanes arbitrary

Exact FP program pairs and numerical dependencies are recorded in the [vector_addition_custom contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
