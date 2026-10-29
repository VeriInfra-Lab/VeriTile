# nested_loops_processing

kernel_correctness

- [nested_loops_processing_repaired.py](nested_loops_processing_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete selected kernel-stage source with all nested loops, cached tile reads, duplicate-lane suppression, clamps, barriers and the entire observed output allocation.

## Conditions

positive: 0<rows*cols and IO.initialized exact signed-int32 input/zero-output snapshots are retained in ImplementsOnCells. Admission allows zero dimensions, but Python does not launch the kernel when n_elements=0; this endpoint is the launched kernel stage, not an empty-host protocol. No floating numerical domain exists.

Exact FP program pairs and numerical dependencies are recorded in the [nested_loops_processing contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
