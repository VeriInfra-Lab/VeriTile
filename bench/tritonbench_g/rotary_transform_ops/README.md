# rotary_transform_ops

Current repaired Python implementation: [rotary_transform_ops.py](rotary_transform_ops.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Explicit conjugated/ordinary rotary prefix a*co-b*signed or a*signed+b*co, with identity coefficient fallback; suffix equals initial input.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
