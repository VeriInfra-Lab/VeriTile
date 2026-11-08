# bmm_chunk_fwd

Current repaired Python implementation: [bmm_chunk_fwd.py](../bmm_chunk_fwd.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Full logical-K A*B-transpose per returned chunk: signed sequence mismatch, padded row/column positions and the causal upper triangle yield zero.

The complete function bodies live in [Sources/Body.lean](../Kernels.lean), with six exact Python-to-Lean pairs recorded in [source correspondence inventory](../../../../validation/source-correspondence.json). The pasted bodies retain all 19 `tl.int64` argument casts, the nested causal return, the signed K loop, both load casts, the `-1`/`-2` sequence sentinels, the output cast and the ten-statement cleanup. Pointer dtypes and call bindings are supplied outside the pasted body.

The original repaired Python and the existing index-commuted FP candidate are unchanged. [candidate-provenance.json](../../../../validation/tasks/bmm_chunk_fwd/candidate-provenance.json) records their hashes and the separately materialized Python source for the retained dot-first mathematical helper. That helper changes only the accumulator addition order and is not the public FP candidate. The actual host domain remains FP32 inputs/output with optional signed INT64 sequence IDs; the existing typed mathematical specializations remain available without narrowing their domains.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
