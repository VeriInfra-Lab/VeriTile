# matrix_vector_multip

fun (matrix,vector) row => ∑ col : Fin M, logicalRow matrix (pid*BN+row) col * vector col

- [matrix_vector_multip_repaired.py](matrix_vector_multip_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Real headline is one complete source invocation and its BN output tile masked by pid*BN+i<N, with full M loop. FP headline independently compares entire N-output grid schedules. This scope difference is explicitly retained, not misrepresented as full-grid Real correctness.

## Conditions

Public conditions:

- 0 < BM for the Real invocation
- FP: 0<BN, 0<BM, Graph BN BM and order.Perm of complete ceil(N/BN) grid
- Shared typed IO binding/bounds and output row mask

Contract relation:

InputContract retains original rank/dtype/device/shape/extent checks; KernelConfig has all 48 configurations; KernelMemory/KernelContract give physical placement and finite program-ID binding without desired output premises.

Exact FP program pairs and numerical dependencies are recorded in the [matrix_vector_multip contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
