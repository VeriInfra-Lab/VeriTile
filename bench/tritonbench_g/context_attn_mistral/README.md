# context_attn_mistral

Current repaired Python implementation: [context_attn_mistral.py](context_attn_mistral.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_real: complete original ordinary grid implements finiteSentinelAttention on the original metadata domain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
