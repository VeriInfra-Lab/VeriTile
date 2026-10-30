# index_select_cat

Current repaired Python implementation: [index_select_cat.py](index_select_cat.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

index_select_cat.kernel_correct: logicalGatherIO implements (i,j) ↦ xs.payload[i,j].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
