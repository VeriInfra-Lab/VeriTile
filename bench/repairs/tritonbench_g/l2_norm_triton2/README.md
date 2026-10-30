# l2_norm_triton2

Forward finite L2 normalization and backward finite q/dot derivative matrices.

- [l2_norm_triton2_repaired.py](l2_norm_triton2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both genuine source kernels retain their complete row grids, all logical output cells, runtime epsilon ABI and exterior frames.

## Conditions

- Each Real stage preserves its explicit N≤B cover premise.
- No epsilon positivity/finiteness restriction is introduced into total Real arithmetic.
- The backward Contract preserves equal x/dy element count and DY last-axis grouping, distinct from l2_norm_bwd's longer-prefix allowance.

Exact FP program pairs and numerical dependencies are recorded in the [l2_norm_triton2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
