# rms_norm_triton

Current repaired Python implementation: [rms_norm_triton.py](rms_norm_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

xs row col * (1 / sqrt ((sum k:Fin N, xs row k * xs row k)/N + eps)) * weights col.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
