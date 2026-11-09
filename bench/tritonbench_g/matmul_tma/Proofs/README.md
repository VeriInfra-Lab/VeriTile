# matmul_tma

Current repaired Python implementation: [matmul_tma.py](../matmul_tma.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fun (../a,b) row col => ∑ k : Fin inner, a row k * b k col

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The Lean quotations copy the complete Python function bodies. FP16 input/output storage, the FP32 output route, all nine runtime dimensions/strides and the constexpr launch arguments are supplied outside the quotations. The direct-output candidate inlines the original optional FP16 conversion into the store; the FP32 source has no explicit cast to remove. Its full output-memory equivalence uses the same native conversion instruction on all words and retains the dot schedules, access safety and exterior frames. Private register formats need not coincide.
