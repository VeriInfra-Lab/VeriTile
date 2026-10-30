# chunk_bwd_dqkg

Current repaired Python implementation: [chunk_bwd_dqkg.py](chunk_bwd_dqkg.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Independent dQ/dK contractions over all logical V with causal 64-token score terms and exponential gates; dG is the per-key-tile BK contraction with the terminal chunk correction. All three formulas are displayed together.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
