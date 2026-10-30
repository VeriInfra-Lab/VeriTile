# fast_ce_loss

Current repaired Python implementation: [fast_ce_loss.py](fast_ce_loss.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Full-vocabulary log-sum-exp/loss, active-chunk LSE and target term, and saved-LSE input gradient with scaling/softcap derivative.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
