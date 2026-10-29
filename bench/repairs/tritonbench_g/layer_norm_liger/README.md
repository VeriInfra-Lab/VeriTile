# layer_norm_liger

Independent finite population statistics, DX expression and exact original Ico row sums for DW/DB.

- [layer_norm_liger_repaired.py](layer_norm_liger_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

The original complete per-program scopes: forward Y/Mean/RSTD; backward every DX in [pid*rowsPerProgram,min((pid+1)*rowsPerProgram,rows)) and both partial parameter-gradient vectors, including empty groups.

## Conditions

- Forward preserves columns≤block and the original pairwise output region separation.
- Backward preserves Logical.FreshDX, DX/DW/DB separation, columns≤block and columns≤dxStride.
- Original IO masks/strides and empty-loop behavior remain unchanged; no full-grid host aggregation is newly claimed.

Exact FP program pairs and numerical dependencies are recorded in the [layer_norm_liger contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
