# bgmv_expand_slice

Current repaired Python implementation: [bgmv_expand_slice.py](bgmv_expand_slice.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each active batch/output row, sum over logical K of X[b,k]*gatheredW[b,n,k], plus the initial Y[b,n] exactly when ADD_INPUTS is true. Signed -1 rows are outside the output write mask.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
