# softmax_flaggems

Current repaired Python implementation: [softmax_flaggems.py](softmax_flaggems.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Forward: mass(q[o])/sum_{n<d.N} mass(q[row,n,col]), mass(-∞)=0 and mass(x)=exp(x). Backward saved-output VJP: p[o]*(g[o]-sum_{n<d.N} p[row,n,col]*g[row,n,col]).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
