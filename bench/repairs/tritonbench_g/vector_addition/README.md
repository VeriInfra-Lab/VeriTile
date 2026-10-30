# vector_addition

Direct logical pointwise Real arithmetic.

- [vector_addition_repaired.py](vector_addition_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One canonical actual FP32 CTA, block size 1024, mask pid*1024+i<n; ordinary regional IO admits overlapping read-only input views and exact exterior frame.

## Conditions

- Regional masked IO supplies actual input binding, active access safety/window bounds; no Real finite restriction
- Contract reuses checked contiguous CUDA FP32 same-device/same-shape/rank-1 domain, extent n<2^31, no negative views and alignment
- FP additionSite requires finite loaded operands only at active stored lanes, separate from Python input admission

Exact FP program pairs and numerical dependencies are recorded in the [vector_addition contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
