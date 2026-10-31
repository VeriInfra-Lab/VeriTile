# attn_fwd_triton

Current repaired Python implementation: [attn_fwd_triton.py](attn_fwd_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Causal base-2 attention: sum over n=0..m of exp(log(2)*score(m,n))*V(n,d), divided by the sum of those weights. score is the complete headDim Q/K dot times the original packed query-block and key-block scales.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
