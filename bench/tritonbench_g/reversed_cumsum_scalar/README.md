# reversed_cumsum_scalar

Current repaired Python implementation: [reversed_cumsum_scalar.py](reversed_cumsum_scalar.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For every t:Fin T, sum k:Fin T of if t.val ≤ k.val then xs k else 0.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
