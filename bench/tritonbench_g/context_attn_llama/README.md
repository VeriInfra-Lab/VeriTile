# context_attn_llama

Current repaired Python implementation: [context_attn_llama.py](context_attn_llama.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.score/numerator/denominator/expected display paged finite-sentinel base-2 normalized attention over all visited padded key lanes; mathematical maps original grid/lane coordinates and gathered logical inputs.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
