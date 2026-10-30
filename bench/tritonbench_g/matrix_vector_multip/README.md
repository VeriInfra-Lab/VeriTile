# matrix_vector_multip

Current repaired Python implementation: [matrix_vector_multip.py](matrix_vector_multip.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun (matrix,vector) row => ∑ col : Fin M, logicalRow matrix (pid*BN+row) col * vector col

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
