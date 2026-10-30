# geglu_tanh_triton

Current repaired Python implementation: [geglu_tanh_triton.py](geglu_tanh_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

geglu_kernel_correctness: logicalIO implements geluValues ch a[i] b[i] (incoming[i] in backward, otherwise zero).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
