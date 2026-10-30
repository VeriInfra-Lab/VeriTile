# ksoftmax_triton

Current repaired Python implementation: [ksoftmax_triton.py](ksoftmax_triton.py).

[Lean source](Kernels.lean) · [input contract](Contract.lean) · [Real correctness](Correct.lean) · [FP equivalence](FPEquiv.lean).

forward_kernel_real_raw: softmax/log-softmax from logical-K partition with exact mask behavior. backward_kernel_real_raw and log_backward_kernel_real_raw: corresponding finite-K derivatives, causal zeroing and extended-value rules.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](Proofs/)

[Source and mathematical adapters](Sources/)

[Selected-task and source-correspondence records](../../../validation/) track validation and pending work separately.
