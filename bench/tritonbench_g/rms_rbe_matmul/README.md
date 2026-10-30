# rms_rbe_matmul

Current repaired Python implementation: [rms_rbe_matmul.py](rms_rbe_matmul.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Adjacent normalizedProjection and projectedRotary mathematics, indexed by logical batch/row/column; optional rotary and the source storage decoder stay explicit.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
