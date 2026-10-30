# rotary_emb_nopad

Current repaired Python implementation: [rotary_emb_nopad.py](rotary_emb_nopad.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

(a*c-b*s, a*s+b*c), explicitly for Q-first/Q-second/K-first/K-second; cached K values are written into the metadata-selected cache.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
