# bgmv_expand_slice

For each active batch/output row, sum over logical K of X[b,k]*gatheredW[b,n,k], plus the initial Y[b,n] exactly when ADD_INPUTS is true. Signed -1 rows are outside the output write mask.

- [bgmv_expand_slice_physical_tile_checked.py](bgmv_expand_slice_physical_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete batch/output-row schedule, both EVEN_K and ADD_INPUTS branches; all logical K terms and all output rows, including empty calls and the real unmasked EVEN_K read width.

## Conditions

Public conditions:

- Call retains BK positivity, rankCovered, evenExact, sliceInside and selected storage metadata
- PublicIO.Valid: each signed index is -1 or nonnegative
- The exact gathered read windows are bound/backed through ProgramIO; Valid does not silently add an adapter-count restriction
- Output mask, InBounds, execution success, safety and exterior frame are preserved

Contract domain source:

Contract.python_checks_iff records the actual tensor/scalar/layout/separation/alignment/static-tile checks. Contract.call, block_covered and selected_valid connect admitted host metadata to Call and valid signed indices. PublicIO.inputWidth preserves BK reads in EVEN_K, including K=0.

Exact FP program pairs and numerical dependencies are recorded in the [bgmv_expand_slice contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
