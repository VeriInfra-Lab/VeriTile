# int8_matmul_quantization

producer_correct: signed8 toward-zero(input/rowScale), rowScale=maxAbs(row)/127. gemm_correct: sum over Fin K of signed8 A*B, then row and column scales.

- [int8_matmul_quantization_i32_safe.py](int8_matmul_quantization_i32_safe.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both required stages retained: every producer row with both source loops and both outputs; every GEMM logical output under complete grouped split invocations, every original scale/output precision and B layout, independent per-cell commit order and typed zero reset.

## Conditions

- Producer.Selection, p.columns≤p.width and original complete row Schedule; no finite-payload restriction.
- GEMM SignedInputs and typed initialValues (0:ℝ) are explicit in ImplementsResetOnCells; original Shape/Config/Plans and complete Schedule remain.
- Checked contract retains active K≤131071 (unless M or N is empty); unchanged original integer/native model and configuration tree bounds.

Exact FP program pairs and numerical dependencies are recorded in the [int8_matmul_quantization contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
