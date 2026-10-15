# Public source and typed execution checkpoints

Current review: [浮点值语义修复与验收结果](ACCEPTANCE.zh-CN.md).
The current bundle is [acceptance-after-value-repair](acceptance-after-value-repair/receipt.json).
The original 41 mathematical-only tasks have 118 additional public contracts:
107 parametric contracts and 11 fixed witnesses. Per-operation backend semantics,
software bit interpretation, accepted stage scope and native bindings are kept
separate in the [repair table](../../tritonbench_g/fp_value_repairs.md).
The remaining gaps are 23 cast/store-only models, three refuted unrestricted
rounding identities and six missing source pairs. Existing relative equivalence
and experimental premises retain their original scope. None of these counts
by itself establishes native functional correctness.
See [remaining gaps](../../tritonbench_g/fp_proof_gaps.md) and [184 task statuses](acceptance-tasks.md).
Earlier reports and snapshots are preserved below; their counts describe their own inputs.

These are local evidence snapshots. Source comparison, Lean elaboration,
native compilation and independent acceptance remain distinct claims.

| Checkpoint | Result | Scope |
|---|---|---|
| [acceptance-after-value-repair](acceptance-after-value-repair/receipt.json) | See frozen gate outcomes and input archive | Current value-contract repair and remaining native/model gaps |
| [acceptance-after-model-review](acceptance-after-model-review/receipt.json) | 6503 Lake jobs; 4087 stable inputs; FP-contract presence fails | Historical model-level inventory before the 41-task repair |
| [acceptance-after-numerical-review](acceptance-after-numerical-review/receipt.json) | 6503 Lake jobs; 4069 unchanged inputs; numerical review gate passes | Superseded numerical-law-only counts; see model review correction; independent replay remains blocked |
| [acceptance-after-parallel](acceptance-after-parallel/receipt.json) | 6503 Lake jobs; 3942 unchanged inputs; all eight libraries pass | Prior complete registered build, exact contracts and source checks; independent replay and six missing source pairs still prevent whole-corpus acceptance |
| [acceptance-current](acceptance-current/README.md) | 3,931 unchanged inputs; source/inventory/contracts pass; full acceptance fails | Exact whole-tree archive, all 184 records, complete gate outcomes, build failure and unavailable official isolation |
| [acceptance-repaired-root](acceptance-repaired-root/README.md) | 3,857 Lake jobs; 25 requested targets; 568 unchanged inputs; exit 0 | Eight pairs of complete Correct/FPEquiv entrypoints, two real-only entrypoints, three legacy modules and all registered frontend regressions; selected local build only |
| [acceptance-branch-real-root](acceptance-branch-real-root/README.md) | 273 unchanged inputs; exit 0 | Shared gated branch execution/safety bridge and dependent chunk GLA public entries |
| [acceptance-constant-branch-root](acceptance-constant-branch-root/README.md) | Six regression targets; 92 unchanged inputs; exit 0 | Both constant branch arms, loops, precision and negative guard tests |
| [acceptance-runner-tests](acceptance-runner-tests/README.md) | Seven unit tests passed; real comparator integration blocked | Missing/stale/failed-gate rejection, actual systemd/Landlock preflight and complete diagnostics |
| [acceptance-changes](acceptance-changes/README.md) | 56 changed files; patch matches current files in reverse check | This turn's edits relative to saved working-tree files, preserving unrelated pre-existing changes |
| [acceptance-integration-failed](acceptance-integration-failed/README.md) | 3,798 Lake jobs; 510 unchanged inputs; exit 1 | SSM and other repaired closures passed; recorded context FP and then-unrepaired gated/GLA failures |
| [acceptance-all-libraries-initial-failed](acceptance-all-libraries-initial-failed/README.md) | Full initial build failed; long remaining job interrupted | Exact baseline diagnostics, including omitted legacy membership at that time |
| [acceptance-repair-stage2-failed](acceptance-repair-stage2-failed/README.md) | Legacy/selected split integration failed | Complete legacy membership exposed additional failures; parallel_attention was interrupted and remains unverified |
| [legacy-public-cleanup-root](legacy-public-cleanup-root/README.md) | 5,197 Lake jobs; 1,841 unchanged inputs; public-entry audit passed | Superseded public entries retired across 37 ports; 86 retained correctness statements unchanged; no new proof coverage or independent acceptance |
| [legacy-public-cleanup-source](legacy-public-cleanup-source/README.md) | Complete 178-view source audit; 3,563 unchanged inputs | Refreshed coverage reviews, manifests, specification sheets and checked inventory after cleanup |
| [multinomial-cast-aggregate-root](multinomial-cast-aggregate-root/README.md) | 4,922 Lake jobs; 1,566 unchanged inputs | Final shared frontend semantics, four storage profiles and registered signed/RNG/rounded-boundary regressions; aggregate proof closures |
| [multinomial-four-profile-stage](multinomial-four-profile-stage/README.md) | Four full-body profile gates; 35 unchanged inputs | fp16/bf16/fp32/fp64 score storage, original i64 buffers and checked constexpr bindings |
| [multinomial-cast-source-root](multinomial-cast-source-root/README.md) | Complete 178-view source audit; 3,595 unchanged inputs | Current reviewed adapter fingerprints, coverage metadata and checked inventory; 49 regression tests recorded separately |
| [multinomial-cast-context-stage](multinomial-cast-context-stage/README.md) | Six Lean regression files; 68 unchanged inputs | Signed narrowing/Bool conversion, nested RNG casts, address inference and full multinomial body admission |
| [multinomial-signed-rng-root](multinomial-signed-rng-root/README.md) | Three axiom-clean theorems; 46 unchanged inputs | Actual signed narrowing discharges RNG offset bounds for arbitrary input integers and supports typed vector RNG execution |
| [multinomial-rounded-boundary-root](multinomial-rounded-boundary-root/README.md) | Six axiom-clean theorems; nine unchanged inputs | A canonical RNG sample lies in a rounded interval overlap; a scalar value-contract obligation, not complete kernel execution |
| [multinomial-native-original](multinomial-native-original/README.md) | 8/8 offline compilations; 40 verified artifacts | Unchanged original JIT, four score dtypes and i32/i64 runtime scalar ABI, original i64 buffers and wrapper block constants |
| [public-barrier-aggregate-root](public-barrier-aggregate-root/README.md) | 4,899 Lake jobs; 1,543 unchanged inputs | Shared debug-barrier semantics, frontend regressions, aggregate libraries and softmax/retention/isfinite public migrations |
| [isfinite-complete-flat-root](isfinite-complete-flat-root/README.md) | 3,431 Lake jobs; 144 unchanged inputs | Complete fp32/fp64 original/candidate typed-flat executions, bit-value formulas, tagged frames and concrete nonempty witnesses; 129 explicit flat-chain axiom gates |
| [isfinite-ordinary-source-root](isfinite-ordinary-source-root/README.md) | 101 Python regressions and full source-only audit | Strict finite-helper dispatch and unchanged arithmetic/extern/order audit behavior |
| [complete-public-source-root](complete-public-source-root/README.md) | Final source-only corpus audit; 3,595 unchanged inputs | All 178 source views, current public contracts, coverage reviews and inventory metadata |
| [isfinite-typed-flat-witness-stage](isfinite-typed-flat-witness-stage/README.md) | 133 unchanged inputs | Concrete allocations, both branches, fourteen special-value inputs and signaling-NaN padding |

The earlier typed-flat component, complete-stage and public-component root
snapshots retain the intermediate evidence and exact sources. The dedicated
`retention-public-source`, `softmax-forward-public-source` and
`isfinite-public-source-pair` directories retain the corresponding strict AST
checks and mutation controls.

The refreshed inventory contains 184 tasks and 178 imported source pairs;
six tasks lack imported sources. It records 629 public entry records after
removing 68 superseded records. Entry records are keyed by file and declaration
spelling; the namespace-aware Lean audit checks the actual declarations.

Historical descriptive coverage aggregation groups 101 tasks under specialized
public statements, 16 under partial statements, one under a blocked statement,
and 60 under full-value labels awaiting current acceptance. Removing obsolete
entries changes these grouped labels without proving additional tasks. The
cleanup preserves all remaining statement scopes and does not change an
independent acceptance ledger. These figures are neither completed-task counts nor missing-numerical-experiment counts; current numerical status is reported above.

Native boundaries remain explicit: retention has 432 compiled and 432 rejected
unchanged-source matrix cases; isfinite has 48 unchanged-original rejections
at the block-pointer `type.element_ty` expression. Local typed-bit correctness
does not establish native libdevice refinement, host-wide launch coverage or
fixed-width scalar/address behavior.

Multinomial remains one of the six ports without a public source pair and
complete sampling contract. Its native source compilation and complete-body
frontend regression establish prerequisites. The rounded boundary and signed
RNG theorems record additional obligations and proved expression behavior;
they do not promote the port's coverage label. The initial root cast aggregate
and the first adapter audit (which stopped for missing rg on PATH) remain
archived with their limitations stated in their own READMEs.
