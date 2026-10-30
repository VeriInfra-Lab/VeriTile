# embedding_triton_kernel

Current repaired Python implementation: [embedding_triton_kernel.py](embedding_triton_kernel.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

For each row/feature, gather weights[id-first, feature] when first≤id<last, otherwise zero.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
