# Original-program scope review

The requested standard covers **all declared stages, dtypes and branches of
183 original programs**. The excluded task is `fp4_to_bf16_conversion`, whose
existing contract is for a repaired candidate. This exclusion does not cover
the separate original-source task `fp4_to_bf16`.

[`original_program_scope.json`](original_program_scope.json) has **2 source scopes accepted and 181 pending**, bound to the current extension
catalog. `matrix_transpose` is verified under explicit primitive contracts without
numerical deferrals; `relu_triton_kernel` is verified with exactly registered
numerical deferrals. Official comparator and whole-corpus
CI results remain separate. It preserves initial gaps
as provenance and records reviewed partial extensions with their remaining
structural work. Its observed input hashes and partial-progress records do not
constitute whole-program reviews. Existing mathematics, equivalence and value
proofs remain available as evidence for each substantive review.

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
[`deferred_numerical_experiments.json`](tritonbench_g/deferred_numerical_experiments.json)
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

The initial exact numerical adapter added 18 unvalidated scalar families to the
11 legacy rules. Its 86 dtype/port bindings cover 58 cases in three earlier declarations:
the two fp64 vector-addition grid contracts and the ReLU access-grid contract.
That earlier ReLU declaration is registered only for N=1025, block_size=1024,
sm90. The independently reviewed finite ReLU extension adds 12 families and 84
exact scalar ports for its full source-outcome contract. The current queue has
41 rules, with 170 exact bindings across 398 cases in four declarations; the
finite contract has 340 cases, including error and unsafe outcomes that use no
numerical callbacks. These registrations do not apply to unrelated declarations.
The checker derives case-specific callback requirements from pinned sources and
requires every applicable scalar-port reference. These registrations discharge
no structural obligations by themselves; acceptance requires the separate
complete source-scope review.

After preparing a new reviewed catalog and rebinding the ledger, run:

```sh
python3 -B bench/check_original_program_scope.py \
  --root . --catalog /path/to/current-source-catalog.json \
  --ledger bench/original_program_scope.json --output /tmp/original-scope-result.json
```

The initial ledger intentionally fails this command. Update a task only after
resolving its substantive gaps and reviewing the exact resulting statements;
changing statuses or copying current hashes is insufficient.

`current_source_audit.py` is the current-layout global runner. It requires exact
catalog regeneration, whole-task-file structural checks, this scope gate,
unchanged historical selected-source comparisons, value/deferred metadata,
official comparator preflight and library replay. The workflow uses this runner
and `audit_trust.sh --catalog …`, which schedules every catalog file including
helpers and extra public contracts. The previous command without `--catalog`
remains available for historical layouts.

The value-metadata adapter runs the complete frozen checker in its pinned
archive tree, then regenerates the current catalog and checks every supplemental
public statement. Supplemental execution declarations require explicit named
stages, dtype profiles and a root review hash of `value_scope(task)`; identities,
frames and strict specializations cannot count as complete execution stages.
Frozen descriptive mistakes use `metadata_corrections` with an exact field path,
before value, after value and reason, bound by the root review. The catalog keeps
the frozen `value_registry_row` and exposes the reviewed correction separately.
The initially identified Conv2d annotation describes fp16 storage incorrectly:
the existing `fp16=true` contract uses fp32 storage and internal half casts.
Its correction does not establish the separate `fp16=false` branch or host program.

The workflow's catalog and supplement now point to the
`softmax-source-host-grid` batch, with the unchanged
`original-scope-extensions/metadata-baseline` and preceding final-source manifest
retained as its historical base. Neither global success nor scope
success establishes corpus comparator replay, the integration build or native
GPU correctness; those remain independent checks.

The full-scope review covers all 183 original programs, with no unreviewed
programs. Matrix transpose has passed its schema2 source-scope check. ReLU has passed
with exactly registered numerical deferrals; 181 scopes remain pending. This is not a claim that whole-corpus CI or official comparator
replay has passed. The cumulative supplemental proof inventory contains 54
execution contracts and
six protocol contracts across 16 tasks; those counts are not completed programs.

The latest partial extensions compose vector addition's 189 existing typed
routes through source-order host allocation/assertions and canonical rejection
cases, and RotaryEmb's original real-erased source through the complete Q-derived
grid with initial-memory formulas and tail frames. Vector safe-view/address and
exact numerical registration gaps remain; rotary rounded execution and dtype/
admission/address-domain gaps remain. Neither task is counted as accepted.

ReLU now has complete finite source coverage for all 17 canonical CUDA profiles,
340 outcome cases and the safe index domain through N=2^31. Its 12 numerical
families / 84 exact ports remain unvalidated. Source pointer/view permissions,
fresh allocation metadata and pid0-only stores are explicit. For N>1024 this
accepts the original physical prefix behavior, with arbitrary untouched output
tail; it does not assert whole-tensor mathematical ReLU or native GPU correctness.

Softmax now composes its original forward/backward real-erased execution across
all four grids, host metadata and arbitrary finite autotune trials, preserving
first compiler rejection and saved-output/gradient aliasing. Its two new public
contracts remain partial progress: typed rounded arithmetic and finite scalar/
pointer/view domains still prevent whole-source acceptance. No additional
numerical registrations or completed programs are counted.
