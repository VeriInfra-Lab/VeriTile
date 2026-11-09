# matmul_dequantize_int4

Current repaired Python implementation: [matmul_dequantize_int4.py](../matmul_dequantize_int4.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The complete original and existing group-extent candidate bodies are pasted in
[Sources/Body.lean](../Kernels.lean), with invocation types outside the bodies.
The candidate [Python program](../matmul_dequantize_int4.py) retains its
original constexpr N and local extent precomputation for all 17 configurations;
[candidate provenance](../../../../validation/tasks/matmul_dequantize_int4/candidate-provenance.json) records the exact change.
The selected native carry-dot graph is derived by
[Sources/NativeLowering.lean](Sources/NativeLowering.lean) under the existing
compiler realization contracts. It is not another source implementation or an
assumed floating-point identity. Independent replay remains required for completion.
