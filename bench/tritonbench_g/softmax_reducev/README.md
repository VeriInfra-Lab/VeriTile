# softmax_reducev

Current repaired Python implementation: [softmax_reducev.py](softmax_reducev.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For batch b/head h/dimension d: sum_{k<counts[b]} mass(logit[b,h,k])*value[b,h,k,d] / sum_{k<counts[b]} mass(logit[b,h,k]), with mass(-∞)=0.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
