# int4_matmul

Current repaired Python implementation: [int4_matmul.py](int4_matmul.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

matmul_grid_real: logicalIO implements the finite logical-K dequantized sum A[row,k]*(Q[k,col]-Z[group(k),col])*scale[group(k),col] on matrixDomain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
