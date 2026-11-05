# softmax_reducev

Current repaired Python implementation: [softmax_reducev.py](../softmax_reducev.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

For batch b/head h/dimension d: sum_{k<counts[b]} mass(../logit[b,h,k])*value[b,h,k,d] / sum_{k<counts[b]} mass(../logit[b,h,k]), with mass(../-∞)=0.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

Both existing JIT bodies in the same Python file are copied statement for statement into `Kernels.lean`; pointer dtypes and runtime/constexpr bindings live outside the quotations. The selected host ABI uses FP32 arrays and signed-i32 metadata. The source has no explicit casts on qk, v or the final accumulator; its FP32 loads and arithmetic are retained.

The general logical IO keeps signed metadata words and the complete signed page-address calculation, including counts greater than maximum and negative page indices. Lengths match the mathematical counts and active starts are nonnegative; no maximum-length or nonnegative-page guard is added to this general interface. The optional checked host interface retains its narrower original allocation conditions. Actual signed-loop execution, FP32 masked loads, full access safety, complete grid weighted sums, output-address scheduling equivalence and exterior frames are proved from the copied bodies. Local builds and source checks do not establish completion; the validation progress record requires current independent proof replay.
