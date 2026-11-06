# matmul_dequantize

Current repaired Python implementation: [matmul_dequantize.py](../matmul_dequantize.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The complete original and existing scalar-address candidate JIT bodies are pasted
in [Sources/Body.lean](../Kernels.lean), with invocation metadata outside the
bodies. The original standalone dequantizer shares the identical
[repaired standalone source](../../matmul_dequant_int4/Kernels.lean).
Both existing Python files are unchanged. The S2 source retains its complete
`SPLIT_K` loop expression and both store/atomic branches. The existing autotune
configuration selects `SPLIT_K=1`; the proof-side iteration count is derived from
`cdiv(K, BLOCK_SIZE_K)`, rather than supplied as an extra source argument.
Independent proof replay remains required for completion.
