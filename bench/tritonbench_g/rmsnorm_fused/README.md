# rmsnorm_fused

Current repaired Python implementation: [rmsnorm_fused.py](rmsnorm_fused.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

xs row col * sqrt ((sum j:Fin width, (xs row j)^2)/width + epsilon)^(-1) * weights col.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
