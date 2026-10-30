# layer_norm_ops

Current repaired Python implementation: [layer_norm_ops.py](layer_norm_ops.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Independent LN/RMS/residual/recompute/bias/parameter-gradient formulas with explicit named multi-output observations.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
