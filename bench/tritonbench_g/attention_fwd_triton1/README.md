# attention_fwd_triton1

Current repaired Python implementation: [attention_fwd_triton1.py](attention_fwd_triton1.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Chunk-causal O(h,row,col)=1/sqrt(D)*sum u<min((row/32+1)*32,T) sum e<D Q*K*V; pre-update H(h,chunk,row,col)=sum u<min(chunk*32,T) K*V.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
