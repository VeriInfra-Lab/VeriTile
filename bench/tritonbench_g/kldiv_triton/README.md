# kldiv_triton

Current repaired Python implementation: [kldiv_triton.py](kldiv_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

forward: probability-zero-extended KL point losses or per-chunk Fin B sums with the original N tail mask. backward: -exp(target) or -target.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
