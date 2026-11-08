# matmul_dequant_int4

Current repaired Python implementation: [matmul_dequant_int4.py](../matmul_dequant_int4.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The complete original and existing unit-stride standalone bodies are pasted in
[Sources/Body.lean](../Kernels.lean), with storage/parameter metadata outside
the source. The unlaunched fused JIT is byte-identical in both Python files to
the [shared repaired matmul source](../../matmul_dequantize_int4/Kernels.lean)
and is registered directly by the correspondence manifest. The original and
existing candidate Python files are unchanged. Independent proof replay remains
required for completion, including the complete half/single grids and frames.
