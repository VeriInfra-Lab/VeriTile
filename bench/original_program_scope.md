# Original-program scope review

The current target is **Real, FP32 and BF16 for 183 original programs, plus
source-required types**, under the approved [primary-version policy](review-checkpoints/2026-10-16/native-mul2-relu-custom-add-example-acceptance/primary-version-policy.md).
Existing wider proofs remain available. This integration changes the legacy
full-source ledger; it does not establish completion of the three primary versions.
The [new three-task applicability inventory](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/primary-version-applicability.md)
records source-case evidence and leaves version completion pending.

The active [scope ledger](original_program_scope.json.gz) records **11 legacy
full-source scopes accepted and 172 pending**. `matrix_transpose` is verified
under explicit primitive contracts. Ten tasks retain registered numerical
deferrals: `quantize_global`, `vector_addition`, `quant_transpose_kernel`,
`triton_mul2`, `relu_triton_kernel`, `vector_addition_custom`, `add_example`,
`add_value`, `sin_kernel` and `cosine_compute`. `fp4_to_bf16_conversion` remains
the separate repaired candidate; the original `fp4_to_bf16` stays in the requested set.

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


The [compressed numerical queue](tritonbench_g/deferred_numerical_experiments.json.gz)
contains **273 rules, 2,943 exact bindings, 22,875 cases and 15 declarations**.
The prior 239 rules, 2,366 bindings and twelve callback tables retain their exact
contents. Every numerical port remains bound to its actual source case. In
particular, cosine host arithmetic has obligations on post-allocation paths;
empty or masked labels do not generically erase host numerical requirements.
Native scalar conformance remains deferred.

The [current catalog](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/current-source-catalog.json.gz) contains 184 tasks,
964 public declarations, 3,690 task modules and 4,121 registered modules.
The [supplement](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/catalog-supplement.json.gz) contains 65 execution and six
protocol contracts. These are inventory counts, separate from primary-version
completion and legacy acceptance.

The workflow uses the current gzip catalog, supplement and active ledger.
The [combined source review](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/root-review.json.gz) and
[recorded integration attempts](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/README.md) retain exact evidence.
The global scope-completeness gate still reports 172 pending tasks. Build,
test, comparator and numerical outcomes are reported only from actual runs;
no GPU or whole-corpus success follows from these counts.

The [previous scope document](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/evidence/baseline-progress.md) and
[previous development summary](review-checkpoints/2026-10-16/native-add-value-sin-kernel-cosine-acceptance/evidence/baseline-current-source-acceptance.md)
are archived byte-for-byte. All earlier checkpoints, wider proofs, reopenings,
numerical registrations and failure records are retained. This batch authorizes
no historical cleanup or deletion.
