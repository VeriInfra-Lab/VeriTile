# fused_recurrent_delta

Current repaired Python implementation: [fused_recurrent_delta.py](fused_recurrent_delta.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fused_recurrent_delta_forward_correctness implements output, residual and finalState; fused_recurrent_delta_backward_correctness implements dq, dk, dv, vectorDBeta, scalarDBeta and dh0 on savedDomain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
