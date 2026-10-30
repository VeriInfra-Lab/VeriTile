# seeded_dropout

Current repaired Python implementation: [seeded_dropout.py](seeded_dropout.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

At logical flat offset i: if canonical Philox-10 randReal(seed,i,bits32)>p then x/(1-p), else zero.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
