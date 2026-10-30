# kldiv_ops

Current repaired Python implementation: [kldiv_ops.py](kldiv_ops.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

forward: pointwise KL loss when red=0, otherwise an unnormalized Fin cols row sum; only raw-target branch clamps to runtime epsilon. backward: -exp(target) in log-target mode, otherwise -target.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
