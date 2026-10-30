# rotary_transform

Current repaired Python implementation: [rotary_transform.py](rotary_transform.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each logical output: rotated prefix a*co-b*signed or a*signed+b*co, optional conjugation; identity coefficient fallback; unchanged input suffix.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
