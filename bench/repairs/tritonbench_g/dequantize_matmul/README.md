# dequantize_matmul

kernel_correct: the complete K-by-N dequantization grid implements weights[row,col] * scales[col].

- [dequantize_matmul_repaired.py](dequantize_matmul_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected two-dimensional INT8-to-FP32 dequantization grid. The separate host torch.mm service is outside this JIT kernel theorem and is not asserted here.

## Conditions

Conditions:

- hK : 0<p.BK
- hN : 0<p.BN
- KernelGrid.Separate p
- PublicIO.SignedBytes explicit input domain
- Canonical IO retains full scale/input/output windows, launch masks and exterior frame

Domain source:

IO PublicIO.sourceIO/result/SignedBytes and Proofs/IOGrid. Contract preserves the larger checked wrapper API without adding its host service to this headline.

Exact FP program pairs and numerical dependencies are recorded in the [dequantize_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
