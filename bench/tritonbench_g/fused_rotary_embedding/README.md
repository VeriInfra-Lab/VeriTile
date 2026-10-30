# fused_rotary_embedding

Current repaired Python implementation: [fused_rotary_embedding.py](fused_rotary_embedding.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fused_rotary_embedding_tile_checked.fused_rotary_real: rotaryIO implements Q/K rotation and V copy on logicalMetadataDomain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
