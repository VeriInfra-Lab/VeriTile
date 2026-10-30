# rms_matmul_rbe

Current repaired Python implementation: [rms_matmul_rbe.py](rms_matmul_rbe.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

Independent adjacent normalizedProjection and projectedRotary definitions; displayed logical tensor coordinates and phase/launch row and column mapping.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
