# uniform_sampling

Current repaired Python implementation: [uniform_sampling.py](uniform_sampling.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Canonical Philox10 word, existing rounded uniform decoder, then the independent lower + u*(upper−lower) affine formula.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
