# context_attn_bloom

Current repaired Python implementation: [context_attn_bloom.py](context_attn_bloom.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.score displays logical-D QK and finite -100000000 causal sentinel; numerator/denominator show all visited padded lanes; expected is the guarded normalized-block recurrence, not an ordinary whole-length normalized attention quotient.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
