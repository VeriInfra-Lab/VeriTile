# bmm_chunk_bwd

Current repaired Python implementation: [bmm_chunk_bwd.py](bmm_chunk_bwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each actual output lane, the finite Dout-transpose times A sum over the true shortened chunk length, plus the optional initial residual.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
