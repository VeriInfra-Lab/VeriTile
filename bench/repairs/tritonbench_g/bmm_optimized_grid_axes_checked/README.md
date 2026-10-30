# bmm_optimized

For each logical batch/row/column, sum k=0..shape.k-1 of A[batch,row,k]*B[batch,k,column]. Logical k is independent of all block/tile dimensions.

- [bmm_optimized_grid_axes_checked.py](bmm_optimized_grid_axes_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All outputs of a complete three-axis source grid under any permutation of programs; all eight original mask branches/configurations, K=0 and empty outputs remain.

## Conditions

Public conditions:

- ps.Perm (programs s c)
- Shape and Config retain source dimensions and the original tuning choices
- ProgramIO exact input/output channels, input binding, InBounds, successful actual source executions, all outputs and exterior frame remain

Contract domain source:

Contract.InputContract extends the original checked domain with exact live grid-axis bounds; empty-domain and axes_exact theorems preserve the existing branch distinction. No new scalar/domain restriction is introduced by this pass.

Exact FP program pairs and numerical dependencies are recorded in the [bmm_optimized contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
