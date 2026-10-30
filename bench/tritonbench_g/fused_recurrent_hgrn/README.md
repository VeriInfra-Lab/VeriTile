# fused_recurrent_hgrn

Current repaired Python implementation: [fused_recurrent_hgrn.py](fused_recurrent_hgrn.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_correct: full forward/backward logicalIO implementation conjoined with the exact directional derivative of the adjacent scalar loss objective, using the same dx/dg.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
