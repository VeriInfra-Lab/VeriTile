# attn_fwd_triton

Causal base-2 attention: sum over n=0..m of exp(log(2)*score(m,n))*V(n,d), divided by the sum of those weights. score is the complete headDim Q/K dot times the original packed query-block and key-block scales.

- [attn_fwd_triton_tile_checked.py](attn_fwd_triton_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every batch/head/query/dimension output of the complete original stage=3 grid; five independently strided input roles and their permitted aliases. This is the independent Real interpretation, not native rounded equality.

## Conditions

Public conditions:

- p.stage=3
- OutputLayout.NonOverlapping p
- Parameters and FlatAlloc retain the actual region/base/stride metadata
- AlgorithmProgramIO.InputBinding and InBounds retain logical inputs and all actual access bounds; its Implements judgment includes success, safety, outputs and exact frame

Contract domain source:

Contract.lean inherits Checked.launchChecks plus the exact live static-tile bound; RealIO and Proofs.RealBindings connect the logical channels to the strided source. Stage=3 and output nonoverlap remain explicit in whole_real.

Exact FP program pairs and numerical dependencies are recorded in the [attn_fwd_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
