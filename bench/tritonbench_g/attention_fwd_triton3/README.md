# attention_fwd_triton3

Current repaired Python implementation: [attention_fwd_triton3.py](attention_fwd_triton3.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Full signed-window attention row maximum and weighted denominator/numerator, seeded by INIT or entry carry. Continuation returns M/L/O; END returns M+log2(L) and O/L with L store masked.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
