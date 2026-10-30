# int_scaled_matmul

Current repaired Python implementation: [int_scaled_matmul.py](int_scaled_matmul.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

int_matmul_correct: signed32(sum_k signed8(A[row,k])*signed8(B[k,col])). int_scaled_matmul_real: signed32(towardZero(Real(signedDot)*rowScale[row])).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
