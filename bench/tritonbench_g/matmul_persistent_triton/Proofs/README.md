# matmul_persistent_triton

Current repaired Python implementation: [matmul_persistent_triton.py](../matmul_persistent_triton.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fun (../a,b) row col => ∑ k : Fin K, a row k * b k col

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The two source quotations in `Kernels.lean` copy the complete repaired
`matmul_kernel_persistent` Python bodies (16 top-level statements each), including
the nine int64 argument casts, Boolean ceiling additions, nested signed loops,
pointer dtype branch, output conversion, masks and store. Call metadata stays
outside the body. The existing candidate changes only the order of the exact
integer multiplication used to form `start_m`.

The source frontends accept distinct integer-argument tuple self-casts and
specialize immutable caller pointer dtype equality even inside loops. They reject
swaps, duplicate tuple targets, other cast widths, pointer rebinding and unknown
dtype queries. Regression theorems check complete execution, safety and actual
typed stores for both FP16 and FP8.

Proofs track the source's signed runtime registers and loop indices. The Real
specification is the independent full-K matrix product. FP equivalence compares
the complete actual source pair, including unsuccessful executions, and then
connects successful safe persistent launches to outputs and exterior memory
frames. The integer rewrite adds no floating numerical assumption. Local builds
and body correspondence do not certify completion; exact-source independent
proof replay and the recorded public theorem/specification gates are required.
