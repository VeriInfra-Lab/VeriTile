# iv_dependent_matmul

Current repaired Python implementation: [iv_dependent_matmul.py](iv_dependent_matmul.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_real: C[row,col] = sum over Fin g.inner of A[row,k]*B[k,col].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
