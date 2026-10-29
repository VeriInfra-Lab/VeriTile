# int4_matmul

matmul_grid_real: logicalIO implements the finite logical-K dequantized sum A[row,k]*(Q[k,col]-Z[group(k),col])*scale[group(k),col] on matrixDomain.

- [int4_matmul_repaired.py](int4_matmul_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete m*n matrix over logical K, all 36 original tile configurations, grouped CTA grid, masked tails, split-K and independent original per-address lane serializations. No tile-width sum is substituted for the full K range.

## Conditions

Public conditions:

- Raw.Geometry x
- tile member of Host.sourceTiles
- lanes.Perm (rawLanes shape tile)
- IO.domain retains signed INT32 packed words and zero-initialized in-place output boundary
- ImplementsOnCells retains exact input cells, all-source success/safety, locality/coverage and raw exterior frame

Contract and domain correspondence:

Contract packed/group/matrix shape and raw geometry match independent Spec.mathematical/Formula.mathematical. The original output reset is a real input-cell boundary and remains explicit; it is not weakened to ordinary Implements. Real arithmetic uses ideal mathematical operands, while FP theorem permits all modeled floating words.

Exact FP program pairs and numerical dependencies are recorded in the [int4_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
