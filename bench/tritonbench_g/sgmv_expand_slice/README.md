# sgmv_expand_slice

Current repaired Python implementation: [sgmv_expand_slice.py](sgmv_expand_slice.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Ordered segment fold: each active output is sum_{k<inner} X[row,k]*W[adapter,column-slice,k], optionally plus previous output; inactive/untouched cells retain previous value.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
