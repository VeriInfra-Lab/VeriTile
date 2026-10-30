# layer_norm_welfold

Mean = ∑j:Fin D X[s,j]/D; population variance and sqrt inverse; affine Y.

- [layer_norm_welfold_repaired.py](layer_norm_welfold_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete S-row grid over the original streaming two-pass source. All D columns are covered by chunk loops, so no spurious D≤block restriction is added. Empty row grids and exact exterior frames remain supported.

## Conditions

- Configuration c retains block=1024 or 2048 and its reduction topology; Real proof retains c.tree.Complete.
- Original Separation X/W/Bias/Y/Mean/Rstd remains explicit.
- Contract retains FP32 CUDA0 metadata, logical/affine shapes, contiguous stride/offset checks, positive D, extents/alignment. General theorem is not narrowed to finite-only payloads.

Exact FP program pairs and numerical dependencies are recorded in the [layer_norm_welfold contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
