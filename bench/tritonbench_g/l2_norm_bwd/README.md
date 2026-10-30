# l2_norm_bwd

Current repaired Python implementation: [l2_norm_bwd.py](l2_norm_bwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

dx[row,col] = dy[row,col]/sqrt(q) - x[row,col]*dot/(q*sqrt(q)), q = ∑j x[row,j]^2+eps

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
