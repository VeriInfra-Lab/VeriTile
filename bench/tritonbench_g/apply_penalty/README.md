# apply_penalty

Current repaired Python implementation: [apply_penalty.py](apply_penalty.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

(if 0 < xs.logits j then xs.logits j / xs.repetition else xs.logits j * xs.repetition) - xs.counts j * xs.frequency - xs.presence

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
