# bmm_optimized

Current repaired Python implementation: [bmm_optimized.py](bmm_optimized.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each logical batch/row/column, sum k=0..shape.k-1 of A[batch,row,k]*B[batch,k,column]. Logical k is independent of all block/tile dimensions.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
