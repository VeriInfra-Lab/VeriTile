# reversed_cumsum

Current repaired Python implementation: [reversed_cumsum.py](../reversed_cumsum.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fun xs time feature => sum t : Fin h.T, if time <= t then xs t feature else 0

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
