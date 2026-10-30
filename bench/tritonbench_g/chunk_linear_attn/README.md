# chunk_linear_attn

Current repaired Python implementation: [chunk_linear_attn.py](chunk_linear_attn.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Mathematics.state/output/finalState show independent causal attention sums; adjoint/gradients show complete VJP sums; result routes all six mathematical output channels.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
