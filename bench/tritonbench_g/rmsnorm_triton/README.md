# rmsnorm_triton

Current repaired Python implementation: [rmsnorm_triton.py](rmsnorm_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

(xs batch row k * sqrt ((sum j:Fin width, xs batch row j * xs batch row j)/width + eps)^(-1)) * weights k.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
