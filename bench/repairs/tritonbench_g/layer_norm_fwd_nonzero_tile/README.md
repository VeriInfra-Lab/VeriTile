# layer_norm_fwd

Same-reader independent row scaling, canonical Philox dropout, ordered residual/RMS/LN and two affine formulas over named tensor inputs.

- [layer_norm_fwd_nonzero_tile.py](layer_norm_fwd_nonzero_tile.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

The complete selected-positive-tile ordinary M-row grid and all seven active output channels Y/Y1/mean/rstd/residualOut/keep1/keep2, including empty M and zero-column statistics.

## Conditions

- widthCheck N=true, Domain.ReadWriteSeparated f r and NativeGeometry.Geometry f r stride M N remain public.
- widthCheck is exactly N≤tile N. tile=max 1 originalTile remains positive/bounded and its arange legality is already proved in Contract.
- All flag-dependent IO masks, seed/rowscale second streams beginning at M, aliases and output windows are unchanged.

Exact FP program pairs and numerical dependencies are recorded in the [layer_norm_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
