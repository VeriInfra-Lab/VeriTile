# attention_score

Current repaired Python implementation: [attention_score.py](attention_score.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

N-term signed-window-filtered base-two attention score sum using supplied normalizer, with exact complement/regular window semantics.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.

The source retains its explicit signed integer conversions, including the signed32 K descriptor offset. The Real initialization and grid proofs use the existing public launch bound to prove that descriptor offset exact; the empty-input return still needs no launch bound. Symbolic FP execution retains the actual descriptor conversion for unrestricted metadata. The scalar-left comparison candidate changes only score-scaling multiplication order.
