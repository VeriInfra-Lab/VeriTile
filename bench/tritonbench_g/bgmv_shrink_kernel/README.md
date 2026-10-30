# bgmv_shrink_kernel

Current repaired Python implementation: [bgmv_shrink_kernel.py](bgmv_shrink_kernel.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Initial Y plus scaling times the complete logical K gathered GEMV; signed index -1 preserves initial Y. Existing toFloat/ofFloat wrappers retain the declared channel/storage interpretation.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
