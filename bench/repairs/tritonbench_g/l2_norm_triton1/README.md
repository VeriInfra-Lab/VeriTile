# l2_norm_triton1

fun x row col => x row col / sqrt(∑j:Fin N,(x row j)^2+eps)

- [l2_norm_triton1_repaired.py](l2_norm_triton1_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

The complete actual dense row grid, all N logical columns, typed output values and full exterior frame; epsilon stays a runtime scalar on every invocation.

## Conditions

- N≤B remains explicit for connecting padded reductions to the logical sum.
- Real sqrt/division are total and eps is unrestricted in the mathematical theorem.
- Contract keeps the original FP32/CUDA/contiguous/nonempty-last-axis/width/alignment domain; the general theorem does not impose extra numeric restrictions.

Exact FP program pairs and numerical dependencies are recorded in the [l2_norm_triton1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
