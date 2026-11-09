# attention_kernel_aligned

Current repaired Python implementation: [attention_kernel_aligned.py](../attention_kernel_aligned.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Full logical attention softmax weighted value sum in both original FP16 and BF16 branches.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` directly quotes the four complete 27-statement repaired and
existing early-query Python bodies. FP16 and BF16 keep their separate dot
programs: FP16 requests a half dot result, while BF16 requests FP32 and
explicitly converts the result. `jit-correspondence.json` checks each body and
its fixed `OUT_DTYPE` invocation separately. Runtime strides and FP32 scale,
pointer storage and constexprs stay outside the copied bodies. No cast or
boundary tuple is added to the copied source.

The optional scalar-infinity frontend supplies the implicit half/BF16
conversion in `tl.where(..., qk, float(../"-inf"))`, following the scalar promotion
rules in [Triton 3.7.1](https://github.com/triton-lang/triton/blob/v3.7.1/docs/python-api/triton-semantics.rst).
It uses established pointer/register types and rejects unsupported control or
unknown tensor types. Strong FP32 operands retain their type. The reference
also retains the actual FP32 runtime-scale arithmetic. `SourceTypes.lean`
contains only pure parameter metadata; existing source APIs call the same
quotations. Complete Real/FP proofs, execution, output/exterior frame and
backed active access safety are verified separately. Local source/build checks
are not independent acceptance receipts; current-source replay is tracked in
`validation/`.
