# fused_layernorm_triton

Current repaired Python implementation: [fused_layernorm_triton.py](fused_layernorm_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

kernel_real: logicalIO implements all-D mean, variance, reciprocal sqrt(variance+1e-5), and affine normalization over named logical tensors.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
