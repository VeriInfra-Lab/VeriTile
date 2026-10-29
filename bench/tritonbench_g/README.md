# TritonBench-G v1 ports

Workspace for VeriTile's port of [TritonBench-G v1](https://github.com/thunlp/TritonBench/tree/main/data/TritonBench_G_v1) — 184 GitHub-scraped real Triton kernels released as the headline channel of TritonBench (ACL 2025 Findings).

This directory holds one subdirectory per kernel. Imported ports bundle **upstream Python source + VeriTile DSL port + a per-kernel README**. The current source catalog distinguishes shared-DSL ports, task-local SC models and repaired candidates; see the current review below.

## Layout

```
bench/tritonbench_g/
├── README.md                       (this file)   # coverage table: ../tritonbench_coverage.md
└── <kernel_name>/
    ├── README.md                   per-kernel notes (status, gotchas, TODO)
    ├── <kernel_name>.py            upstream Python source (pinned, see Provenance)
    ├── Kernels.lean                original and candidate Triton sources
    ├── Correct.lean                real IO contracts and mathematical formulas
    ├── FPEquiv.lean                admitted-atom FP equivalence and domain
    ├── FPCorrect.lean              typed operation/value or software-bit contracts
    └── Proofs/                     execution and supporting lemmas
```

Current source and public-declaration discovery uses
[`bench/current_source_catalog.py`](../current_source_catalog.py), including
`FPCorrect`, `SourceCorrect`, `ConcurrentCorrect` and `ProtocolCorrect` files.
The historical layout scanner remains available for replaying its own snapshots.

## Current floating-point review

Installed classification: **184 tasks, 58 software bit-value profiles, 125 backend-dependent profiles and one repaired candidate**. The original 41 remains **35 software / 6 backend-dependent**. There are **893 public declarations**: 747 historical plus 146 new (142 value, two frame, one identity, one protocol).

The [Retention source extension](../review-checkpoints/2026-10-15/retention-query-source-proof/README.md) adds full fp32 DQ and sequential DQ/barrier/DK/DV helper contracts after index specialization. Its 20 named axiom checks passed; numerical bindings remain in the [deferred experiment queue](deferred_numerical_experiments.md). No additional software bit-closed task or GPU success is claimed.

The preceding destindex_copy and matrix_transpose contracts prove complete original typed bit values from independent initial memory, including accepted aliases and full frames. Existing scopes and output-address premises remain; fixed witnesses do not substitute for parameterized proofs. See the [current review/table](../review-checkpoints/2026-10-15/raw-copy-software-metadata/README.md), [value ledger](fp_value_current_status.md) and [reproduction guide](../current_source_acceptance.md). Build evidence combines 3,826 revalidated records with eight independent compilations and 44 named axiom checks.

The [previous dequant acceptance](../review-checkpoints/2026-10-15/dequant-int4-source-compiled-software-metadata/README.md) and [native capture package](../review-checkpoints/2026-10-15/dequant-int4-native-characterization/README.md) retain their own scopes; GPU execution remains NOT_RUN. Official comparator replay targets frozen F3's 365 files / 872 headlines, excluding 21 later declarations. No full-batch success or live progress is asserted. The [daily index](../review-checkpoints/2026-10-15/README.md) links frozen results, experiments and remaining work. Historical workflow counts below are unchanged.

## Historical port workflow

A port goes through three stages, tracked per-kernel in `README.md`:

1. **DSL port** — `<KernelName>.lean` follows the **faithful 1:1 transcription** contract for the upstream `.py` kernel in `triton { ... }` syntax. Allowed mechanical Lean-syntax changes are documented in [`review_criteria.md`](./review_criteria.md); explicit translation exceptions are registered in [`completion_audit.md`](./completion_audit.md#remaining-blockers). The port may not compile if it uses DSL surface that has not yet landed — failing-to-compile is the intended signal that the DSL surface needs extension. **Inventory: 178 Python/Lean port pairs; 6 of the 184 work directories remain source-import scaffolds.** Staged proof helpers do not count as imported ports. Compilation and audit evidence is described below.
2. **Spec** — Real-valued mathematical specification of the kernel's intended output is written.
3. **Verification** — `ComputeCorrect.Realizes` / `ComputeRefine.Realizes` theorem is proved and registered in `scripts/kernel-manifest.tsv`.

Stage 1 is the verbatim transcription contract; reaching stage 3 (verification) requires both the DSL gap to close and a proof to land.

## Historical 41-task audit snapshot

The [checked inventory](current_completion_inventory.md) lists all 184 tasks,
178 imported source pairs and 747 public entry records. These are inventory
counts, not completed-task counts.

The original 41 mathematical-only tasks now have 118 additional public value
contracts in their explicitly accepted configuration and stage scopes. The
[value-contract review](fp_value_repairs.md) distinguishes backend-parametric
contracts, software bit interpretation and fixed witnesses. Their primitive and
native bindings remain pending in the [per-task validation list](fp_value_native_tasks.json).
An empty extra-law list does not establish GPU correctness.

The [remaining model gaps](fp_proof_gaps.md) retain 23 cast/store-only tasks,
three tasks with a refuted unrestricted rounding identity, and six missing
source pairs. Existing relative equivalence results and reports retain their
own conditions; they do not establish independent native value correctness.

Every proof module in the repaired 41 task directories is included in the
registered library builds. The [historical review](../review-checkpoints/2026-10-14/ACCEPTANCE.zh-CN.md)
records actual build and gate results with frozen inputs. Historical receipts
certify their own snapshots. Official independent replay also requires the
comparator's host isolation support.

```sh
python3 bench/fp_value_repairs.py --check
python3 bench/numerical_obligations.py --check
python3 bench/inventory_tritonbench_g_completion.py --check
python3 bench/acceptance_tritonbench_g.py --output /tmp/tritonbench-review
python3 bench/acceptance_tritonbench_g.py --check /tmp/tritonbench-review
```

The bundle includes all task records, an input archive, gate logs and hashes.
The acceptance commands return failure for missing sources, failed/skipped gates
or stale inputs. Passing mechanical checks does not enlarge theorem domains,
prove native refinement, or discharge pending experimental bindings.

## Historical audit notes (earlier snapshots)

The following notes describe earlier source/proof snapshots. Their wording and
historical replay counts must not be read as acceptance of the current files;
several older temporary evidence paths are no longer available.

The [block-padding review](block_padding_review.md) reopened three earlier
accepted cases because omitted padding was translated as zero. Scalar and
vector cumsum have now been repaired and independently replayed, including
proofs that their valid real outputs are independent of arbitrary tail values.
Dqkg is also repaired and replayed: its real contract retains zero unspecified
values, and a full-source counterexample shows why that condition is necessary.
Chunk delta now also has complete source contracts and a checked counterexample
showing its zero-unspecified-value premise is necessary.
The historical acceptance ledger recorded 59 cases and 125 pending. That count
has not been re-established for the current tree. The four older attention
cases reopened for padding-source alignment now pass the complete repair,
72-entry rebuild and independent replay; see `verification_round.md`.
Earlier audit snapshots do not certify later source corrections.
`chunk_gla_fwd` now has all five complete source/candidate pairs, ten real
contracts, five conditional FP equivalences and a full-source padding
counterexample. Its 82-file independent replay is accepted.
`chunk_gla_simple` also passes its complete source/real/FP review and 13-file
replay. Its real formula covers every K block, and its FP proof uses only
`mul_commute`. `chunk_linear_attn` now also passes: all four complete source
pairs, eight real flat-memory contracts and four conditional FP equivalences,
with all 19 case/regression files independently replayed. `chunk_retention`
also passes all four source/real/FP pairs and its 20-file independent replay.
`chunk_retention_ops` also passes all four pairs, including its distinct
reverse-state recurrence, three-chunk regression and 10-file replay.
`chunked_cumsum_fwd` also passes its full finite/default-infinite clamp, real
flat IO and FP review, with 35-file replay and a six-file follow-up for clamp
dtype checking. `context_attn_bloom` now also passes its complete real flat IO
and admitted FP review, typed/signed source checks and 68-file independent
replay. Its real formula retains the finite causal sentinel and accumulator
guard. `context_attn_fwd` now also passes both real flat IO contracts, admitted
FP equivalence, accumulator-dot regressions and 71-file independent replay.
Llama and Mistral context attention now also pass their complete real/FP
contracts, source checks and joint 97-file independent replay.
Nopad context attention and cosine now also pass complete real/FP contracts,
source checks and joint 59-file independent replay.
Cross entropy now passes both complete JITs, four real specifications, two FP
equivalences and 19-file independent replay.
Scaled cross entropy also passes both complete JITs, the gated z-loss output,
four real specifications, two FP equivalences and 22-file independent replay.
Cross entropy ops also passes both complete JITs, the gated z-loss output,
four real specifications, two FP equivalences and 19-file independent replay.
Decay cumsum also passes all three complete JITs, six real specifications,
three FP equivalences and 27-file independent replay.
Matrix dequantization also passes its complete signed-integer JIT, two real
specifications, FP equivalence and 13-file independent replay.
Rowwise dequantization also passes its complete signed-integer source, two real
specifications, FP equivalence and 10-file independent replay.
Destination-indexed dual copy also passes its complete signed-metadata source,
two real specifications, FP equivalence and 14-file independent replay.
Masked multi-head copy also passes its complete signed-metadata source,
two real specifications, FP equivalence and 13-file independent replay.
Head-masked multi-head copy also passes its complete signed-metadata source,
two real specifications, FP equivalence and 13-file independent replay.
Diagonal SSM also passes all four typed sources, eight real specifications,
four FP equivalences and 50-file independent replay.
Dropout also passes complete typed int32/bool keep sources, two real specifications,
FP equivalence and 19-file independent replay including the frontend change.
Embedding also passes the full signed-ID source, arbitrary chunk loop,
finite flat real contracts, FP equivalence and 16-file independent replay.
Both fp8 conversion kernels now pass four real specifications, two typed FP
equivalences, precision/cast checks and 11-file independent replay.
`fast_ce_loss` now passes three complete source pairs, six real contracts,
three atom-free FP equivalences and 163-file independent replay. Its capped
padding and unwritten chunk outputs are explicitly recorded source behaviors;
ordinary CE and host aggregation correctness are not claimed. `fast_layernorm`,
`fast_rms_layernorm` and `fast_rope_embedding` also pass their full review and
52-file replay; source-specific padding, stride and aliasing behavior is
explicit. `fifth_order_sph_harmonics` is in development. See
[`verification_round.md`](./verification_round.md).

The current sweep is tracked in [`completion_audit.md`](./completion_audit.md).
`bench/check_ports.sh` compiles every Python/Lean port pair and requires
official comparator replay. The aggregate audit also checks comment-stripped
sources for placeholder proofs. Completed CI evidence is pinned to a specific
run and commit in [`coverage-ci.json`](../../site/src/lib/coverage-ci.json);
it does not certify later revisions or pending runs. A `null` record means
no completed public-repository corpus audit has been recorded yet.

There are no current explicit algorithm-layer `hAlg` blockers. Any future
proof blockers should be listed in [`proof_blockers.md`](./proof_blockers.md).
The stronger #2 proof-status audit is tracked in
[`proof_gap_manifest.tsv`](./proof_gap_manifest.tsv) and checked by
[`../check_proof_gap_manifest.py`](../check_proof_gap_manifest.py). That
manifest covers public real-correctness and FP-equivalence declarations across
the 175 imported ports. The review inventory, checked on 2026-10-13, records
257 `full_value_candidate`, 382 `specialization`, 4 `projection_only`,
26 `precomputed_input_slice`, 3 `pre_rounding_slice`, and 2 `blocked_summary`
rows: 674 declarations across 278 public specification files.
[`coverage_review.json`](./coverage_review.json) records each headline's
actual scope and checks it against Lean/Python source fingerprints and Python
function links. Explicit source annotations remain binding; source changes
require a new review. A candidate label does not certify GPU execution, and
historical issue links do not imply that the issue is currently open or that
closing it completed a proof.

## Build

The migrated directories are importable through the separate
`TritonBenchGVerified` Lake library. Unmigrated ports remain standalone. Neither
is part of the default VeriTile library build. To check them:

```bash
# all currently-ported kernels
bench/check_ports.sh

# source/metadata checks without proof replay
bench/audit_tritonbench_g.sh --source-only

# complete source and proof audit
bench/audit_tritonbench_g.sh

# reviewed scope classification for every headline declaration
python3 bench/check_proof_gap_manifest.py

# subset by kernel name
bench/check_ports.sh vector_addition softmax_triton1
```

The port-check script elaborates every selected Lean file, including nested
proofs, and requires official comparator export/replay. It reports per-file
pass/fail and exits non-zero on
any failure. The aggregate audit combines compilation, Lean trust/statement
checks, and comparator replay in `bench/audit_trust.sh`, avoiding a duplicate
port-build pass. The public CI workflow runs the structural and library gates
once, then partitions the standalone corpus across four GitHub-hosted jobs.
The final completion check requires the global job and all four shards to pass.
Local runs cover the whole corpus
unless `AUDIT_TRUST_SHARD_COUNT` and `AUDIT_TRUST_SHARD_INDEX` are explicitly set
(see [`scripts/README.md`](../../scripts/README.md#shared-comparator-gate)).
Its other gates include Python/Lean count matching,
placeholder-proof scanning, correctness-surface scanning, compiled-port README
status checks, and a documented-scope check for Python `.to(tl.float32)` casts
that are outside a Lean proof slice. It also rejects Lean-only
`tl.load(..., dtype=...)` annotations and `keep_dims` reduction substitutions,
and it requires Python `+=` updates missing from Lean to be covered by a
documented slice or branch/surface scope, including a normalized left-hand-side
check for pointer names such as `a_ptr` versus `A`. Upstream `rsqrt` calls are
also checked for preservation, and Lean-only `tl.where` statements are
rejected. The same audit compares Python and Lean `tl.*(...)` call surfaces and
requires any missing or extra call to be covered by an explicit
slice/specialization note, and it compares `for` / `while` / `if` counts inside
the Python `@triton.jit` kernel body and Lean `triton { ... }` body. Ordered
`tl.*(...)` call sequences are checked as well, so unannotated call reordering
is rejected mechanically. Top-level statement left-hand-side sequences are also
checked, including `+=`, annotated assignments, and tuple assignments.
It also checks that `proof_gap_manifest.tsv` is fresh against the Lean source,
so newly added or renamed `output_summary` declarations cannot bypass the #2
proof-status classification.
It is a mechanical gate only; line-by-line faithfulness still follows
[`review_criteria.md`](./review_criteria.md), and unresolved proof obligations
remain tracked in [`proof_blockers.md`](./proof_blockers.md).

## Provenance

| Date imported | Upstream commit | Kernels | Notes |
|---|---|---|---|
| 2026-05-06 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 15 (Tier 1) | initial DSL ports; no specs / theorems yet |
| 2026-05-13 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 141 port pairs | the audited port set as of that date; see `completion_audit.md` for remaining proof obligations |
| 2026-07-04 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 1 (`reversed_cumsum_scalar`) | reverse-range port; DSL blocker resolved by reverse-range support |
| 2026-08-25 | [`603e28a`](https://github.com/thunlp/TritonBench/commit/603e28a) | 173 port pairs | current audited port set (`int_scaled_matmul` closed the integer family); the remaining 11 of the 184 upstream kernels are README-only scaffolds |

### Local modifications to vendored `.py` files

The vendored `.py` files are **not** strictly byte-identical to upstream. The following modifications may be applied locally to imported files:

- **Input type annotations on every `@triton.jit` kernel signature.** Pointer args annotated `tl.tensor`, runtime int scalars `tl.int32`, runtime float scalars `tl.float32`. `tl.constexpr` annotations from upstream are preserved as-is. These annotations are JIT-equivalent (Triton ignores non-`constexpr` Python type hints at compile time), so kernel behavior is unchanged — they exist purely as in-source documentation that aligns Python signatures with the type information the Lean ports rely on.

When importing a new batch:

1. Pin the upstream commit you fetched from in this table.
2. Ensure the upstream LICENSE has not changed (currently **Apache-2.0**).
3. Add per-file attribution headers in each `.py` (see [Licensing](#licensing) below).
4. Apply the input-type annotations described above to each `@triton.jit` signature.

## Licensing

Upstream `thunlp/TritonBench` is licensed under **Apache-2.0**. VeriTile is MIT-licensed; Apache-2.0 → MIT vendoring is permitted with attribution.

Each vendored `.py` file should carry an attribution header similar to:

```python
# Source: thunlp/TritonBench@<commit-hash>
#   data/TritonBench_G_v1/<filename>.py
# Upstream license: Apache-2.0 (see https://github.com/thunlp/TritonBench/blob/main/LICENSE)
```

The initial imports landed without these headers (commit `eab9b81`); back-filling them is open work.

## Adding a kernel

1. Pick a file from upstream `data/TritonBench_G_v1/`.
2. Cross-check its verdict in [`../tritonbench_coverage.md`](../tritonbench_coverage.md). Prefer `portable now`; a `blocked` kernel only after the primitive it names has landed (with a consumer).
3. Create `bench/tritonbench_g/<kernel_name>/`.
4. Drop the upstream `.py` in there with the attribution header above.
5. Add `<KernelName>.lean` with the DSL port. Use `namespace VeriTile.Bench.TritonBenchG.<KernelName>` and import `VeriTile.Triton.Core` + `VeriTile.Triton.DSL`.
6. Verify with `bench/check_ports.sh <kernel_name>`.
7. (Stage 2/3) When you write a spec / proof, add a `scripts/kernel-manifest.tsv` row with `source = tritonbench:<filename>.py` and `source_ref = <upstream-commit>`.
8. Update the Provenance table above if importing from a fresh upstream commit.

## See also

- [`../tritonbench_coverage.md`](../tritonbench_coverage.md) — measured coverage classification across all 184 kernels
- [`../README.md`](../README.md) — overall benchmark policy
- [`../check_ports.sh`](../check_ports.sh) — port build script
- [`../../documents/KernelManifest.md`](../../documents/KernelManifest.md) — manifest schema (used at stage 3)
- [`../../documents/TheoremSurfaces.md`](../../documents/TheoremSurfaces.md) — naming convention for verification theorems
