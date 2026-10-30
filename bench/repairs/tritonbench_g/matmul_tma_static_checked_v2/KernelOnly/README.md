# matmul_tma

fun (a,b) row col => ∑ k : Fin inner, a row k * b k col

- [matmul_tma_static_checked_v2.py](../matmul_tma_static_checked_v2.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete original program whose block dimensions equal the full static matrix shape; both transpose flags and FP16/FP32 output branches. It is not advertised as a multi-program grid.

## Conditions

Public conditions:

- Shape and Flags explicitly determine params/strides
- ProgramIO input binding and InBounds supply actual source safety
- c_injective h f StaticChecked.regions is proved internally

Contract relation:

Selected KernelOnly/Contract retains the five original frontend guards and three <=1048576 block-tensor element limits. The formal source theorem holds on its broader typed Shape model; it does not replace API admission or assert arbitrary hardware tile compilation.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_tma contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
