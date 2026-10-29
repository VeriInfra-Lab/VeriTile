# multinomial_sampling

The original full token loop is now imported through the shared Triton DSL, with backend and software bit-semantics theorems. The declared profile covers fp16/bf16/fp32 and explicitly bound fp64 storage, signed64 seeds, and power-of-two blocks at most 1048576. A real-valued storage tag alone does not establish fp64 correspondence.

Public proof entries: [FPCorrect.lean](FPCorrect.lean).

The [host contracts](HostCorrect.lean) additionally execute the complete
`ceilDiv(batch_size, 8)` grid, with `BLOCK_N=128`, against an independent
initial-memory reference. They account for every output row, partial batch
blocks, and the full memory frame. The logical wrapper models the original
shape checks, `indices=None` arange/expansion (including the singleton-batch
stride), first-column clone,
and the single-token early return. The latter returns **shape `[B, 1]`** with
int64 zeros before inspecting indices; the kernel branch returns **shape
`[B]`**. Empty token dimensions fail at the first-column selection.

These host contracts cover the existing four score-storage profiles and
**signed64 indices**. The Python `indices: Tensor` argument has no dtype check;
other supplied-index dtypes and score-storage profiles have not been established
by these declarations. They remain scope review items, not deferred numerical
assumptions. The clone/default-index allocation is represented by fresh logical
regions; native allocator, typed-pointer/address and launch/concurrency
refinement remain explicit external boundaries. Numerical tree/cast/FTZ
binding obligations of the kernel are unchanged.

Native GPU correspondence remains unestablished. Offline compilation and Lean proofs have separate meanings; current scope and experimental obligations are recorded in the [184-task review](../../review-checkpoints/2026-10-14/next-value-repair/final-acceptance/tasks-184.tsv) and [experiment handoff](../../review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md).

The previous import-status note is preserved in the [documentation history](../../review-checkpoints/2026-10-14/next-value-repair/documentation-update/before/bench/tritonbench_g/multinomial_sampling/README.md).
