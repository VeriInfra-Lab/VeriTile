# fast_ce_loss

Current repaired Python implementation: [fast_ce_loss.py](../fast_ce_loss.py).

The checked entry loads [fast_ce_loss.py](../fast_ce_loss.py). Its three
JIT function bodies are copied directly into `Kernels.lean`, with caller metadata
outside the quotations. [fast_ce_loss.py](../fast_ce_loss.py) retains the
row-index multiplication commutation for each complete stage.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Full-vocabulary log-sum-exp/loss, active-chunk LSE and target term, and saved-LSE input gradient with scaling/softcap derivative.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
