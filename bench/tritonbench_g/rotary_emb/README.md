# rotary_emb

Current repaired Python implementation: [rotary_emb.py](rotary_emb.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each Q/K interleaved pair: even=a*cos_even-b*sin_even, odd=a*sin_odd+b*cos_odd. Four explicit output channels.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
