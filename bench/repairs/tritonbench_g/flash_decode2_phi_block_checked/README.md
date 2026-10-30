# flash_decode2_phi

Exponentially weighted summary average at every feature over each row's logical steps.

- [flash_decode2_phi_block_checked.py](flash_decode2_phi_block_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected flash_decode2_phi_block_checked masked B*H grid, arbitrary complete permutations, all D outputs with D≤BD padded-lane coverage. Logical read capacity is independent of source stride S.

## Conditions

- Real headline explicitly retains Fits and Positive: used count≤capacity and nonempty used count per launched row. Empty batch/head grids remain valid.
- PhiIO.iterations preserves exact source integer behavior at zero length and Q=0; FP pair requires only Fits, not Positive, and admits these counts as well as count>S when backing capacity suffices.
- Contract adds only active-launch block≤2^20 and preserves original empty cases. The conditional Real theorem is not advertised as a total Real specification for every native exceptional case.
- IO retains original S strides, metadata-dependent used-summary masks, D/BD tail masks, initial bounds/bindings and all output/frame obligations.

Exact FP program pairs and numerical dependencies are recorded in the [flash_decode2_phi contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
