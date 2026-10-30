# spinning_lock_reduction

The original Python source is present. A task-local typed AST represents both while loops and all nine rounds; five public declarations cover source correspondence, backend/software schedule semantics and the memory frame. The software model uses fp32 addition and signed-i32 control, arbitrary modeled initial memory/locks and every finite sequentially consistent (SC) schedule, including blocked and fault prefixes. This is a task-local model; shared DSL while/atomic-condition support is not claimed.

Public proof entries: [ConcurrentCorrect.lean](ConcurrentCorrect.lean), [SourceCorrect.lean](SourceCorrect.lean).

The proofs do not establish termination, race freedom or refinement to GPU memory ordering. Initially ready locks are not assumed to justify the complete launch.

Native GPU correspondence remains unestablished. Offline compilation and Lean proofs have separate meanings; current scope and experimental obligations are recorded in the [184-task review](../../review-checkpoints/2026-10-14/next-value-repair/final-acceptance/tasks-184.tsv) and [experiment handoff](../../review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md).

The previous import-status note is preserved in the [documentation history](../../review-checkpoints/2026-10-14/next-value-repair/documentation-update/before/bench/tritonbench_g/spinning_lock_reduction/README.md).
