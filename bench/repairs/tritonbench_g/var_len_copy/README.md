# var_len_copy

Every logical destination lane equals the corresponding initial source lane.

- [var_len_copy_repaired.py](var_len_copy_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete dynamic segment program, all length iterations/tail lanes; all three signed metadata values read even when length≤0, no stores then and arbitrary signed starts retained. Whole independent segment schedule remains optional.

## Conditions

- RegionalImplementsOn3D segmentDomain pins signed length and only pins nonnegative starts for positive a.len
- InputContract has signed int32 metadata, FP32 payload, counts/extents, physical separation, positive-segment source/destination bounds and disjoint destination intervals
- Actual source widens int32 metadata to int64 before loop/address arithmetic; no floating arithmetic or finite-value guard

Exact FP program pairs and numerical dependencies are recorded in the [var_len_copy contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
