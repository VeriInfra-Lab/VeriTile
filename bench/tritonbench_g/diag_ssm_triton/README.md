# diag_ssm_triton

Current repaired Python implementation: [diag_ssm_triton.py](diag_ssm_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Forward time recurrence and reverse carried/input/initial/transition gradients for FP32-real and complex64 mathematical interpretations.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
