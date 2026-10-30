# context_attn_fwd

Current repaired Python implementation: [context_attn_fwd.py](context_attn_fwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.score displays logical-D QK and finite -100000000 sentinel; mathematical displays the full visited-window base-2 exponential weighted numerator divided by the full padded denominator, with decoded original grid/lane coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
