# Original-program scope review

The current development target is **Real, FP32 and BF16 for 183 original
programs, plus source-required types**, as specified by the approved
[primary-version policy](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md). The
[eight-task evidence map](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/three-version-inventory/eight-task-mapping.md) preserves existing proofs and records the
work still needed for each selected version. No new three-version completion
count is assigned by the queue storage migration.

This page describes the retained **legacy full-source scope checker**, whose
contract covers all declared stages, dtypes and branches. Its earlier wider
proofs remain reusable evidence; its counts do not establish completion of the
three selected versions. `fp4_to_bf16_conversion` remains the separate repaired
candidate, and the `fp4_to_bf16` original task remains in the requested set.

The active [`original_program_scope.json.gz`](original_program_scope.json.gz)
records **8 legacy full-source scopes accepted and 175 pending**. `matrix_transpose` is
verified under explicit primitive contracts. `quantize_global`, `vector_addition`,
`quant_transpose_kernel`, `triton_mul2`, `relu_triton_kernel`,
`vector_addition_custom` and `add_example` are verified with exactly registered
numerical deferrals. Unintegrated producer candidates do not change these counts.

The [finite acceptance boundary](original_program_acceptance.md) separates
source-specific obligations from trusted Python/Torch/Triton primitive semantics.
It preserves real coverage requirements without adding universal compiler, runtime
or test-harness implementation proofs. The reviewed first calibration changes
four gap lists and introduced no acceptance by itself.

The checker rejects omitted tasks, pending reviews, changed sources, missing
dimensions, unresolved structural gaps, candidate/standalone proof substitutions,
and missing or stale review artifacts. Each accepted task binds every task Lean
and Python file, every original-source function and conditional/loop inventory,
and a whole-source dtype-domain obligation even when Python contains no literal
dtype token. `test_*` helpers may be classified as test harnesses only by an
explicit review; JIT functions and functions called by production code cannot
use that classification. No function is automatically removed from the ledger.

Every covered obligation needs its full dtype/configuration scope, an explanation
and references to current public proof statements. A separate JSON review with
`kind: original_program_scope_review`, `reviewer: root`, and `status: reviewed`
binds the exact catalog, input hashes, source inventory, coverage, numerical
deferrals and callback mappings. Its `semantic_review` must explain why those
proofs cover the original program. The checker verifies those bindings; a human
or agent substantive review is still responsible for the semantic conclusion.

Accepted rows require `scope_schema: 2`. Each execution declaration's reviewed
`original_scope` binds the source-inventory hash, exact Python AST stage/branch/
dtype-domain IDs and a structured input/output/compute dtype plus configuration
case for every covered obligation. Cases must match the ledger exactly. A whole
dtype-domain obligation requires a declaration explicitly reviewed as covering
the complete original or enumerated wrapper domain; a selected fp32 profile
cannot cover it. Kernel-stage evidence cannot cover a different host stage.
This verifies metadata compatibility, not the truth or exhaustiveness of the
metadata; the theorem-to-source correspondence and exhaustive domain review
remain substantive review duties.

Numerical conformance may remain conditional through
`verified_with_deferred_numerics`. Every used law/model ID and every unclosed
numerical callback must map to an explicit rule in the unified
[`deferred_numerical_experiments.json.gz`](tritonbench_g/deferred_numerical_experiments.json.gz)
queue, with its exact rule hash, dtype profile, configuration and required future
validation. A callback without a registered rule stays blocked. Refuted rules,
missing translations, omitted stages/branches/dtypes, host composition, value
domain restrictions and concurrency/progress gaps cannot be deferred as numerical
conformance. Existing empirical reports retain their profile limitations; this
gate does not convert them into hardware-refinement proofs.

Each rule's `scope_bindings` must also identify the task, exact public statement
hash, callback, structured dtype profile and configuration. Deferrals and
callback links reference both the rule ID and the scope-binding ID. A matching
rule name or arbitrary free-text configuration is insufficient: a fp32 rule
cannot certify fp64, and an exp2 callback cannot borrow a rounding-only binding.

The current compressed JSON numerical queue has **239 rules, 2,366 exact bindings,
22,317 cases and 12 declarations**. Empty, all-masked and rejected cases require
no unused numerical callbacks. Prior 153 rules, 1,642 exact bindings and eight
callback tables remain unchanged. New numerical relations remain unvalidated.

The current catalog contains 184 tasks, 961 public declarations, 3,666 task
modules and 4,097 registered modules. Its supplement contains 68 declarations:
62 execution and six protocol contracts. These inventory counts are separate
from the 8/183 source acceptance count.

The workflow now uses the [2026-10-16/native-mul2-relu-custom-add-example-acceptance](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/README.md) catalog and supplement:

- [Current catalog](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/current-source-catalog.json.gz)
- [Current supplement](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/catalog-supplement.json.gz)
- [Combined source review](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/root-review.json.gz)
- [Integration checks and retained attempts](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/README.md)

The original metadata baseline and preceding final-source manifest stay frozen.
The global audit checks exact catalog reproduction, whole-task structural
coverage, source-scope reviews, legacy selected sources, value metadata, numerical
bindings, comparator preflight and library replay. Its scope-completeness gate
continues to fail for the 175 pending tasks. Build, test and comparator results
are recorded from actual runs in the current checkpoint; this progress document
makes no independent build, comparator, GPU or numerical-conformance claim.

```sh
python3 -B bench/check_original_program_scope.py \
  --root . \
  --catalog bench/review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/current-source-catalog.json.gz \
  --ledger bench/original_program_scope.json.gz \
  --output /tmp/original-scope-result.json
```

## Historical progress snapshots

The [previous scope progress document](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/baseline-progress.md)
and [previous development summary](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/evidence/baseline-current-source-acceptance.md)
retain their exact original bytes. Their older counts, pending conclusions and
catalog paths describe those historical snapshots. Earlier bounded vector/ReLU
contracts, scope reopenings, withdrawn quant-transpose recommendations, partial
extensions and numerical registrations remain in their frozen checkpoints.
The new reviewed contracts supersede the older pending conclusions for the
four tasks accepted in this batch; historical evidence itself is unchanged.

The [pre-migration summary](@MIGRATION_CHECKPOINT@/before/bench/original_program_scope.md) is preserved byte-for-byte with the gzip migration evidence.
