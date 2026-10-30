# token_softmax_bloom

Current repaired Python implementation: [token_softmax_bloom.py](token_softmax_bloom.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fun xs job i => exp(xs.values job i) / sum (j : Fin(xs.length job)), exp(xs.at job j.val)

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
