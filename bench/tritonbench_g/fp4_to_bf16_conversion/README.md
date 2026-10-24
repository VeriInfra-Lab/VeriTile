# fp4_to_bf16_conversion

The shared Triton DSL now represents the repaired scaled-kernel candidate. Its two public theorems cover backend and software bit semantics for packed uint8 e2m1/e8m0 to bf16 with `mx_block_size = 32`. The original Python compilation failure is retained; these theorems establish the candidate contract, and the original-source proof remains missing.

Public proof entries: [FPCorrect.lean](FPCorrect.lean).

Native GPU correspondence remains unestablished. Offline compilation and Lean proofs have separate meanings; current scope and experimental obligations are recorded in the [184-task review](../../review-checkpoints/2026-10-14/next-value-repair/final-acceptance/tasks-184.tsv) and [experiment handoff](../../review-checkpoints/2026-10-14/next-value-repair/native-experiment-handoff/EXPERIMENT-HANDOFF.md).

The previous import-status note is preserved in the [documentation history](../../review-checkpoints/2026-10-14/next-value-repair/documentation-update/before/bench/tritonbench_g/fp4_to_bf16_conversion/README.md).
