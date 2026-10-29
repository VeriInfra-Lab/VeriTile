# matmul_triton1

fun (a,b) row col => ∑ k : Fin innerSize, a row k * b k col

- [matmul_triton1_repaired.py](revision001/matmul_triton1_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete row-major FP32 source grid; full logical K, empty/K=0 behavior, tails and all original configurations retained.

## Conditions

Public conditions:

- FiniteSource.Canonical p
- FiniteIndex.Ready p
- FiniteIndex.StorageBounds p
- Geometry.Valid (Core.geometry p)
- p.input != p.output and p.weight != p.output
- Public ProgramIO binding/bounds and exact exterior frame

Contract relation:

Contract.Launch and Binding retain Tensor metadata, allocation separation, original input admission and launch-width facts. No new Python restriction or result/launch-success premise was added.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_triton1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
