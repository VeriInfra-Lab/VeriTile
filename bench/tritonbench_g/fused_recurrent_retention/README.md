# fused_recurrent_retention

Current repaired Python implementation: [fused_recurrent_retention.py](fused_recurrent_retention.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_forward_real and kernel_backward_real: retentionIO implements independent prefix/suffix formulas indexed by actual (part,head,time,channel) and final (head,key,value) coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
