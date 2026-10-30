# token_attn_llama2

Current repaired Python implementation: [token_attn_llama2.py](token_attn_llama2.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun xs b h t => (sum d, xs.query b h d * xs.keyAt (m.page b t) (h/(heads/kvHeads)) d) / sqrt D

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
