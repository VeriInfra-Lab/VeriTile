# multinomial_sampling

The original full token loop is now imported through the shared Triton DSL, with backend and software bit-semantics theorems. The declared profile covers fp16/bf16/fp32 and explicitly bound fp64 storage, signed64 seeds, and power-of-two blocks at most 1048576. A real-valued storage tag alone does not establish fp64 correspondence.

Public proof entries: [FPCorrect.lean](FPCorrect.lean).

Native GPU correspondence remains unestablished. Offline compilation and Lean proofs have separate meanings; current scope and experimental obligations are recorded in the [184-task review](../../review-checkpoints/2026-10-14/next-value-repair/final-acceptance/tasks-184.tsv) and [experiment handoff](../../review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md).

The previous import-status note is preserved in the [documentation history](../../review-checkpoints/2026-10-14/next-value-repair/documentation-update/before/bench/tritonbench_g/multinomial_sampling/README.md).
