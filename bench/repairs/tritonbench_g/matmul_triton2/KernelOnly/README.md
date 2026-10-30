# matmul_triton2

fun (a,b) row col => ∑ k : Fin innerSize, a row k * b k col

- [matmul_triton2_repaired.py](../revision001/matmul_triton2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected grouped launch grid; strided matrix channels, empty/tail behavior and original accumulator-fed padded TF32 dot graph retained.

## Conditions

Public conditions:

- Geometry.Valid (Core.geometry p)
- 0 < p.blockInner
- input/output and weight/output region separation
- ProgramIO binding and bounds supply all logical outputs, access safety and frame

Contract relation:

Selected KernelOnly/Contract keeps revision001 input admission and existing output shape/layout/device/alignment/disjoint allocation facts; no source or Python contract changed.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_triton2 contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
