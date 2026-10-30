# bmm_chunk_fwd

Current repaired Python implementation: [bmm_chunk_fwd.py](bmm_chunk_fwd.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Full logical-K A*B-transpose per returned chunk: signed sequence mismatch, padded row/column positions and the causal upper triangle yield zero.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
