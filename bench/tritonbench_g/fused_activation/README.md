# fused_activation

Current repaired Python implementation: [fused_activation.py](fused_activation.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

fused_activation_kernel_correctness: activationIO implements j ↦ sigmoid(z) or max 0 z, where z = input[j]/2 + x[j] + bias[j].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
