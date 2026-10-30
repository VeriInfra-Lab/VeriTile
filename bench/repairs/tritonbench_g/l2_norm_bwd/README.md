# l2_norm_bwd

dx[row,col] = dy[row,col]/sqrt(q) - x[row,col]*dot/(q*sqrt(q)), q = ∑j x[row,j]^2+eps

- [l2_norm_bwd_repaired.py](l2_norm_bwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete row grid with independent signed64 X/DY/DX row strides, logical N columns, exact exterior frame and runtime epsilon.

## Conditions

- N≤block is explicit, as is PublicIO.OutputUnique on DX addresses.
- Input windows can overlap; no new input disjointness, epsilon positivity or finite-only assumption is added.
- Contract retains DY last-axis grouping, x.numel divisibility and x.numel≤dy.numel, plus its dtype/device/layout checks.

Exact FP program pairs and numerical dependencies are recorded in the [l2_norm_bwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
