# rmsnorm_fused_llama

Current repaired Python implementation: [rmsnorm_fused_llama.py](rmsnorm_fused_llama.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

xs row col * sqrt ((sum k:Fin columns, (xs row k)^2)/columns + epsilon)^(-1) * weights col.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
