# cache_transform_triton

Current repaired Python implementation: [cache_transform_triton.py](cache_transform_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Every output byte j is payload[rows[j/rowBytes]*rowBytes + j%rowBytes], with signed row metadata interpreted under its public nonnegative/range domain.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
