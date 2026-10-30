# l2_norm_triton1

Current repaired Python implementation: [l2_norm_triton1.py](l2_norm_triton1.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun x row col => x row col / sqrt(∑j:Fin N,(x row j)^2+eps)

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
