# token_attn_mistral

Current repaired Python implementation: [token_attn_mistral.py](token_attn_mistral.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

sum n in range xs.attention(job), if n+max(xs.sequence(job)-window,0)<xs.sequence(job) then probAt xs job n * valueAt xs job n d else 0

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
