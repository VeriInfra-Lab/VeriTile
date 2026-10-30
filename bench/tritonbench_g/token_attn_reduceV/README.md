# token_attn_reduceV

Current repaired Python implementation: [token_attn_reduceV.py](token_attn_reduceV.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun xs b h d => sum n in range (counts b), xs.probAt b h n * xs.valueAt b h n d

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
