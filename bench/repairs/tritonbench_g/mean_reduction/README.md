# mean_reduction

fun xs row => (∑ col : Fin N, xs row col) / (N : ℝ)

- [mean_reduction_repaired.py](mean_reduction_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every row of a complete arbitrary permutation of ceil(M/8) source programs; source row/column streaming masks, tails, empty grid and formal N=0 Real division retained.

## Conditions

Public conditions:

- tiles.Perm (List.range (gridSize M))
- meanIO carries the identical realIO and realSchedule; LogicalScheduledKernelIO.Implements preserves complete regional binding, bounds, execution, outputs and frame.

Contract relation:

Original normalized-axis contract rejects empty reduced axes and checks distinct axes/extents; the formal grid theorem validly includes N=0 without silently imposing a new Python admission. Tensor interpretation remains optional application scope.

Exact FP program pairs and numerical dependencies are recorded in the [mean_reduction contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
