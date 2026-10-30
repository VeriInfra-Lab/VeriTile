# int_scaled_matmul

int_matmul_correct: signed32(sum_k signed8(A[row,k])*signed8(B[k,col])). int_scaled_matmul_real: signed32(towardZero(Real(signedDot)*rowScale[row])).

- [int_scaled_matmul_repaired.py](revision001/int_scaled_matmul_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both unscaled and scaled complete grouped grids; both EVEN_K branches, K=0/empty/tail cases and exact exterior frames. One scale per row; ignored scaleColumnStride remains ignored as in source.

## Conditions

- Geometry.Valid and positive unscaled blockInner; output separation from A and B.
- Scaled EVEN_K exact-step premise, full inner coverage and separation from A/B/scale; all original strides/configurations remain.
- Real scaling/truncation is an ideal mathematical interpretation, independent of the FP all-word graph and undefined-conversion policy.

Exact FP program pairs and numerical dependencies are recorded in the [int_scaled_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
