# rope_backward_transform

Current repaired Python implementation: [rope_backward_transform.py](rope_backward_transform.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Displayed adjoint (a*c+b*s, b*c-a*s) on both logical Q/K outputs; every odd tail column returns the original input.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
