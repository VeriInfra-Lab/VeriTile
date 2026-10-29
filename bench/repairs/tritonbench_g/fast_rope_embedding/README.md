# fast_rope_embedding

Forward or adjoint two-coordinate RoPE rotation at each active head/column.

- [fast_rope_embedding_repaired.py](fast_rope_embedding_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One launched row/head-group including the actual dynamic four-head loop. The existing r.Launched premise, head/column masks and exterior frame preserve odd trailing columns. No whole-tensor strengthening.

## Conditions

- Explicit launch, row*qStride=row*(heads*dimension), cosineStride=half and sineStride=half remain.
- KernelContract.active_stride derives the active Q stride equation from existing metadata and ordinary Tensor row-index law; singleton unused strides remain permitted.
- IO gives actual in-place Q window, coefficient rows, masked bounds/bindings, successful execution and exact exterior frame.

Exact FP program pairs and numerical dependencies are recorded in the [fast_rope_embedding contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
