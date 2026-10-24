# Current development and acceptance

The current goal is **183 original programs with Real, FP32 and BF16 as the
three primary versions, plus source-required types**, under the approved
[primary-version policy](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md). Each selected version retains its actual
stages, branches, layouts, memory behavior and failure paths. The
[eight-task evidence map](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/three-version-inventory/eight-task-mapping.md) records reusable proofs and remaining
version-specific gaps.

The **legacy full-source scope ledger records 8 accepted and 175 pending**.
These counts retain the earlier wider-domain work; they do not mean that eight
tasks have completed all three versions. This storage migration assigns no new
version-completion counts.
`matrix_transpose` is verified under explicit primitive contracts. Seven tasks
are verified with exactly registered numerical deferrals: `quantize_global`,
`vector_addition`, `quant_transpose_kernel`, `triton_mul2`, `relu_triton_kernel`,
`vector_addition_custom` and `add_example`.

The active [scope ledger](original_program_scope.json.gz) and
[acceptance policy and current scope review](original_program_scope.md) define
that count. `fp4_to_bf16_conversion` remains the separate repaired-candidate
entry. Numerical deferrals do not waive missing source, host, dtype, branch,
address, memory or failure-order coverage. Producer candidates awaiting
integration do not count as accepted programs.

The current cumulative integration is [2026-10-16/native-mul2-relu-custom-add-example-acceptance](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/README.md). Its
[current catalog](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/current-source-catalog.json.gz)
contains **184 tasks, 961 public declarations, 3,666 task modules and 4,097
registered modules**. The
[supplement](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/catalog-supplement.json.gz)
has 62 execution and six protocol declarations. These are proof-inventory
counts, separate from original-program acceptance.

The [compressed numerical queue](tritonbench_g/deferred_numerical_experiments.json.gz)
contains 239 rules and 2,366 exact bindings across 22,317 cases in 12 declarations.
The queue remains fail-closed against the reviewed source-derived authority;
these registrations do not establish native scalar conformance. Existing
refutations and all numerical-validation obligations remain preserved.

The workflow selects the current explicit `.json.gz` catalog, supplement and
active scope ledger. It retains the unchanged historical metadata baseline and
final-source review manifest. The current checkpoint's
[validation and attempt records](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/README.md)
report actual builds, tests and global gates. The 175 pending scopes remain a
completeness failure. Historical comparator successes apply only to their
pinned trees; no current comparator, whole-corpus, GPU or numerical result is
inferred from source acceptance or inventory counts.

## Historical progress snapshots

The [previous development summary](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/baseline-current-source-acceptance.md)
and [previous scope progress](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/baseline-progress.md)
are preserved byte-for-byte. Their 0/183 or 4/183 summaries, host-and-LoRA catalog,
previous build/test counts, partial-scope tables and historical comparator
results describe earlier states. Their proof evidence and checkpoint links
remain available; the current summary above supersedes their progress claims.

The [pre-migration summary](@MIGRATION_CHECKPOINT@/before/bench/current_source_acceptance.md) is preserved byte-for-byte with the gzip migration evidence.
