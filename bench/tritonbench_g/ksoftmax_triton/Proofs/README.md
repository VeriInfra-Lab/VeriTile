# ksoftmax_triton

Current repaired Python implementation: [ksoftmax_triton.py](../ksoftmax_triton.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

forward_kernel_real_raw: softmax/log-softmax from logical-K partition with exact mask behavior. backward_kernel_real_raw and log_backward_kernel_real_raw: corresponding finite-K derivatives, causal zeroing and extended-value rules.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Compatibility source imports](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Both complete existing forward/backward Python bodies are copied into `Kernels.lean`, including all three int64 index conversions, optional mask-string guards, both forward/backward IS_FP16 branches, causal repair and original stores without invented output casts. Caller scalar widths and constexpr values remain outside the quotations. The selected host ABI is FP32; the backward compatibility wrapper supplies the corresponding false IS_FP16 flag outside the body.

The optional-string frontend freezes the supplied constexpr value and supports only None guards or equality with literal strings. It rejects writes, loop shadowing and unsupported uses; an independent complete typed-program equality covers absence, recognized strings and other string values. The native compiler metadata records the exact current Python source hash. Local source/build checks never establish completion; the validation progress record requires current independent proof replay.
