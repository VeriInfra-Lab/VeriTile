# int8_matmul_kernel

Current repaired Python implementation: [int8_matmul_kernel.py](int8_matmul_kernel.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_low32: signed low32 of the four packed-field dense integer sum, using the original defined FP32-to-byte cast.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
