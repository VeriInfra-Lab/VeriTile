# Deferred numerical experiments

This is a handoff for later experiments. No GPU experiment was run by this work. The existing software/backend proofs keep their exact scopes.

The inventory covers 184 tasks: 77 have concrete native-binding recipes, and the others retain their relative-equivalence/historical review scope. An empty callback list is not hardware validation.

## Numerical rule profiles

| Profile | Disposition | Affected tasks |
|---|---|---:|
| `fp32_add_commute_real` | `reported_profile_only` | 22 |
| `fp32_mul_commute_real` | `reported_profile_only` | 17 |
| `fp32_div_mul_rcp_real` | `reported_profile_only` | 10 |
| `fp32_mul_commute_typed` | `pending_exact_binding` | 2 |
| `ROUND-FP16-ID` | `refuted_unrestricted_assumption` | 3 |
| `ROUNDING-MODEL-STRUCTURE` | `model_requirement_not_hardware_evidence` | 31 |
| `BGMV-ADD-fp64-real` | `pending_exact_binding` | 1 |
| `BGMV-ADD-fp32-real` | `conditional_on_reported_model` | 1 |
| `BGMV-ADD-fp32-fp16` | `pending_exact_binding` | 1 |
| `BGMV-ADD-fp32-bf16` | `pending_exact_binding` | 1 |
| `BGMV-ADD-fp32-fp32` | `pending_exact_binding` | 1 |
| `VA-F64-N01` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N02` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N03` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N04` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N05` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N06` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N07` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N08` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N09` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N10` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N11` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N12` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N13` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N14` | `exact_profile_bound_validation_pending` | 1 |
| `VA-F64-N15` | `exact_profile_bound_validation_pending` | 1 |
| `RELU-W-N1-FROMINT` | `exact_profile_bound_validation_pending` | 1 |
| `RELU-W-N1-FROMNAT` | `exact_profile_bound_validation_pending` | 1 |
| `RELU-W-N2-LITERAL` | `exact_profile_bound_validation_pending` | 1 |

Five profiles still need exact binding. Existing reports remain limited to their recorded configuration. The unrestricted fp16 rounding-is-identity assumption is refuted; it must be repaired, not submitted as a validation target.

## New Retention proof dependencies

The new DQ and sequential DQ/DK/DV proofs use explicit M/D backend parameters. They prove source execution, complete outputs and the memory frame. They do not identify those parameters with native instructions.

| Obligation | Status | Relation to establish |
|---|---|---|
| `PRA-DQ-EXP2-LOG2` | `experiment_not_run` | Bind M.unary fp32 exp2/log2 at every decay site to the exact compiler/libdevice graph. Head decay is log2(sub32(1,exp2(sub32(-5,mul32(fromNat(head),1))))). |
| `PRA-DQ-DOT-ACCUMULATION` | `compiled_graph_proof_and_experiment_pending` | Source D.apply retains allow_tf32=false and two-argument dot followed by a separate accumulator add. Native live-carry dot/FMA must have its own recurrence; source dot-from-zero plus add is not assumed equal to dotAcc. |
| `PRA-DQ-SCALAR-SCALE-CAST` | `experiment_not_run` | Bind M.binary, M.literal, M.cast, M.fromNat/M.fromInt and fp32Load. Preserve dq*(rowDecay*scale) between loops and (dot*diagonalDecay)*scale inside the diagonal loop, including the source casts. |
| `PRA-DQ-PADDING-UNDEF` | `semantic_refinement_and_experiment_pending` | Boundary loads have no padding_option. The source uses address-indexed s.undef, never an assumed zero. Relate native undefined/poison behavior to that model explicitly. |
| `PRA-DQ-INTEGER-ADDRESS-ABI` | `semantic_refinement_and_experiment_pending` | Relate unbounded Nat addresses and widened signed differences to actual source integer widths, physical byte storage, legal allocations and launch indices. Zero-step source semantics is not a legal native launch guarantee. |
| `PRA-BACKWARD-COMPOSITION-NATIVE` | `semantic_refinement_and_experiment_pending` | The source expansion executes DQ then barrier then DK/DV; later references read post-DQ memory. Relate native helper inlining, barrier behavior, aliases and cross-program order to the source before claiming kernel correctness. |
| `PRA-HOST-NK-NV-SUM` | `outside_proved_scope` | PyTorch sums of partial outputs across NK/NV, dtype conversion, host allocation and autograd dispatch require separate models and proofs. |

## Reproduction and batch handoff

Run from the repository root:

```sh
python3 -B bench/deferred_numerical_experiments.py --check
```

The [JSON queue](deferred_numerical_experiments.json) contains all task scopes, declaration references, binding relations, existing job recipes, corner cases, report records and source hashes. The [Retention supplement](parallel_retention_attention/NUMERICAL_ASSUMPTIONS.json) gives per-obligation captures.

Keep failed and not-run cases. A source/compiled graph mismatch requires a corrected graph or an explicit scoped relation before numerical comparison; do not silently reassociate arithmetic, replace undefined padding with zero, or substitute dotAcc for dot-from-zero plus add.

## Exact scalar follow-up bindings

18 additional scalar families bind 86 exact dtype/port instances across 58 declaration cases. All remain unvalidated.

The fp64 cases bind the frozen sm90 frontend configuration. ReLU binds only N=1025, block_size=1024, sm90; other specializations remain unregistered. The queue preserves structural gaps and adds no accepted original programs.
