# index_select_cat

index_select_cat.kernel_correct: logicalGatherIO implements (i,j) ↦ xs.payload[i,j].

- [index_select_cat_repaired.py](index_select_cat_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete BI*BC source program at explicit rowPid/colPid with both logical row/column tail masks; int64 index metadata, all selected payload values, safe actual accesses and output frame. No whole-grid theorem is newly claimed.

## Conditions

Public conditions:

- LogicalIO.Unique p rowPid colPid ensures distinct active output writes
- Actual index input binding and metadata-dependent source payload read are in gatherIO
- ProgramIO active read/write InBounds and input memory bindings remain

Contract and domain correspondence:

Contract target is source(indices i, d), with valid signed indices, canonical contiguous Layout specialization, native bounds and disjoint output/index/source. LogicalIO.read performs the signed index/stride selection, so copying the payload channel is an independent gather target, not an output/run premise.

Exact FP program pairs and numerical dependencies are recorded in the [index_select_cat contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
