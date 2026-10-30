# index_select_bwd

Current repaired Python implementation: [index_select_bwd.py](index_select_bwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

index_select_bwd_kernel_correctness: indexIO implements i ↦ current[i]+incoming[i].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
