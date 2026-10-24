# TritonBench-G Completion Audit

This document separates the port inventory, the scope of each headline theorem,
and the checks required for a successful audit. Passing the mechanical gates
is not evidence that every upstream kernel has a full correctness proof.
Faithful translation still requires review against
[`review_criteria.md`](./review_criteria.md).

## Latest local integration checkpoint — 2026-10-13

The previous shared and historically registered aggregate passed all 4726 jobs
after conditional casts, builtin zero and register-dtype/where precision
support, with all 1371 recorded Lean/build inputs unchanged during the run.
Attention's explicit mathematical erasure, L2 projection simplification and
RMSNorm's exact fp32 where marker were adapted without removing typed source
statements. The failed intermediate rebuilds and successful repairs are in
[`aggregate-register-where/receipt.json`](../review-checkpoints/2026-10-13/aggregate-register-where/receipt.json).
This aggregate excludes the three in-progress ports; it is not their acceptance.
The later [reverse-loop and pointer-cast repair](../review-checkpoints/2026-10-13/reverse-range-pointer-cast/receipt.json)
passes 15 frontend/interpreter regressions with unchanged recorded inputs.
Descending loops now preserve signed bounds, zero iterations and captured
range operands; known pointer element casts preserve all six floating storage
types and both dynamic dtype binders at the current program point. Both complete
reverse-cumsum sources compile with the actual casts retained. The subsequent
[aggregate revalidation](../review-checkpoints/2026-10-13/aggregate-reverse-pointer/receipt.json)
passes all 4726 jobs with 1371 recorded inputs unchanged. The separate
[completed-port revalidation](../review-checkpoints/2026-10-13/completed-ports-reverse-pointer/receipt.json)
passes all 130 modules: LayerNorm 70, uniform 41 and seeded dropout 19, with all
recorded shared and local sources unchanged.
Their [full typed FP relations](../review-checkpoints/2026-10-13/reverse-full-fp/receipt.json)
also pass for arbitrary algebras and independent input/output types without
floating assumptions. Failed real-proof siblings remain explicitly unaccepted.
The real formulas retain unspecified tail values: ordinary suffix sums need
zero total padding or an exact chunk boundary, rather than an invented zero
padding policy.
The later [complete real executions](../review-checkpoints/2026-10-13/reverse-real-execution/receipt.json)
now pass for both sources and candidates at every time length, including zero,
with independent initial-memory formulas and full tagged-cell frames. Vector
finite IO, trace safety and nonzero-padding execution regressions also pass;
its earlier typed-body sibling failure is recorded separately. The later
[joint finite-IO revalidation](../review-checkpoints/2026-10-13/reverse-finite-io/receipt.json)
passes both ports, including the scalar finite allocation witness and both
nonzero-padding regressions, with all 181 captured inputs unchanged. Vector
complete typed success also passes in the same profiled interpreter as its FP
relation. The public migrations now pass complete 30-module closures for both
[scalar](../review-checkpoints/2026-10-13/reverse-scalar-public/receipt.json) and
[vector](../review-checkpoints/2026-10-13/reverse-vector-public/receipt.json),
with 194 and 193 captured inputs unchanged respectively. Both expose two full
real contracts and one empty-assumption typed FP relation. The vector legacy
single-block results now explicitly require a zero input oracle; their prior
unqualified padding claim was not justified by the unchanged source. The new
public region contracts retain arbitrary oracle values. Coverage review now
records 682 declarations in 285 public files: 269 full-value candidates, 383
specializations, 22 prepared-input slices, four projections, three pre-rounding
slices and one blocker. These labels do not establish independent acceptance. The later
[global source-only audit](../review-checkpoints/2026-10-13/reverse-public-source-audit/receipt.json)
passes with all 3724 recorded inputs unchanged. Its memory-call comparison
now resolves checked pointer-element dtype attributes on both sides, consistent
with the existing cast gate; 71 audit regressions pass, including rejection of
changed output types, values and boundary indices. This audit performs no
independent proof replay.
The two [complete original-source AST checks](../review-checkpoints/2026-10-13/reverse-source-ast/receipt.json)
pass with 14 rejected mutation controls and unchanged inputs. The separate
[native admission matrix](../review-checkpoints/2026-10-13/reverse-native/receipt.json)
compiles all 396 unchanged-source configurations on the offline sm90 target;
all 1584 stored IR hashes were verified. The 84 empty-time cases contain no
Triton IR loads or stores. These are selected shape/ABI observations, not GPU
execution, native numerical refinement or independent formal replay.
The first whole-corpus source-only run passes the 177 selected-source extraction
and expression/control scans, but fails the proof-gap manifest gate on stale
review fingerprints. After reviewing the changed contracts, both coverage and
the refreshed manifest now pass for 682 public declarations in 283 files.
The rerun of the [combined source-only audit](../review-checkpoints/2026-10-13/source-audit-current/receipt.json)
passed with root exit code 0. Its failed predecessor remains separate;
source-only success does not include proof replay.

The [backward softmax type-family checkpoint](../review-checkpoints/2026-10-13/backward-full-fp-and-typed-io/receipt.json)
proves complete typed source/candidate state-and-failure equality for all 216
independent storage triples with no floating assumptions. The homogeneous typed
real IO/frame module also passed; its historical batch failed in the separate
trace-control helper. The later
[current erasure and FP checkpoint](../review-checkpoints/2026-10-13/backward-current-erasure-and-fp/receipt.json)
passes the complete 216-type original-source erasure-to-execution and independent
real output/frame bridge, and revalidates the full typed FP relation after the
frontend repairs. The 96 homogeneous typed executions and trace-control,
trace-probe, variant and LOG value helpers also pass; failed probe predecessors
remain separate. The later
[complete backward real and typed checkpoint](../review-checkpoints/2026-10-13/backward-full-real-and-typed/receipt.json)
passes the original and candidate finite IO contracts, complete active-address
safety, both ordinary and logarithmic causal value witnesses, and the public
proof barrel. Its 199 recursively captured local inputs remain unchanged.
This completes the backward per-program mathematical bridge for all formal
storage triples; forward flag coverage, native profile admission, host launch,
machine integer behavior and independent replay remain separate obligations.
These results do not admit the natively rejected profiles.

The later [LayerNorm public/typed checkpoint](../review-checkpoints/2026-10-13/layernorm-fp32-public-and-typed/receipt.json)
compiles all 70 local modules, including complete real and finite IO contracts
and same-engine successful typed consistency witnesses. The witnesses enable
every optional feature for both normalization modes; they are not IEEE or
native executions. Even selected fp32 storage still needs host scalar ABI
encoding and fixed-width address bridges, as well as independent replay.
The [uniform public and typed checkpoint](../review-checkpoints/2026-10-13/uniform-public-and-typed/receipt.json)
passes all 41 local modules and 29 public axiom checks, including both complete
real/frame/finite IO contracts, arbitrary-algebra typed success and the full
typed FP relation. Public same-engine and finite-allocation witnesses include
four active streams, a partial tail and signed counter wrap without precomputed
random outputs. All 41 Lean inputs remained unchanged during compilation.
The [seeded-dropout adaptation](../review-checkpoints/2026-10-13/seeded-dropout-where-adaptation/receipt.json)
also passes all 19 modules and 24 actual complete-source materializations after
fixing a metadata comment before imports and simplifying the explicit compute
dtype in the existing successful-execution proof.
The [backward typed checkpoint](../review-checkpoints/2026-10-13/backward-typed-execution/receipt.json)
checks 96 complete source executions in an explicit consistency algebra.
The [216 backward source profiles](../review-checkpoints/2026-10-13/backward-source-profiles/receipt.json)
retain all three source flag branches and reject 1728 expression/control
mutations. Native rejection boundaries below remain in force.

The shared aggregate `lake build VeriTile VeriTileFull TritonBenchGVerified
TritonBenchSpecExamples` passed all 4726 jobs after the direct-Philox frontend
and signed fixed-width integer operations were integrated. The recorded targets are those
enumerated when that build started; this does not admit new in-progress ports
to `TritonBenchGVerified`. The earlier failed aggregate below is retained as
history. Logs and the precise evidence boundary are preserved in
[`aggregate-direct-frontend/receipt.json`](../review-checkpoints/2026-10-13/aggregate-direct-frontend/receipt.json).
The earlier 4723-job aggregate and selected-profile source checks remain in
[`aggregate-and-guarded-source/receipt.json`](../review-checkpoints/2026-10-13/aggregate-and-guarded-source/receipt.json).

Separate local builds passed the BGMV conditional FP bundle, the migrated
typed softmax bundle, and the LayerNorm first-stream execution phases. The
complete selected fp32 LayerNorm branch tree passed 24 mechanical source gates;
the softmax selected fp32 source passed 23 Python gates on unchanged captured
inputs. These selected-profile checks do not establish all-profile source
admission. Source checks and local compilation are separate from independent
proof replay and whole-task completion.

The direct-Philox frontend passed 12 axiom checks and 13 rejection checks;
12 affected Lean regressions and 36 existing source-materialization tests
also passed. The reviewed adapter fingerprint and exact local scope are in
[`direct-random-frontend/receipt.json`](../review-checkpoints/2026-10-13/direct-random-frontend/receipt.json).
Uniform-sampling proof helpers, further LayerNorm execution phases and new
backward softmax modules remain separate in-progress work. Their compilation
and source checks do not imply completion of their full contracts.
The [LayerNorm original-source checkpoint](../review-checkpoints/2026-10-13/layernorm-original-real/receipt.json)
now proves successful execution of the complete erased source and an independent
initial-snapshot postcondition for all seven output paths and every other tagged
memory cell, across all ten flags. Its 36 axiom checks passed. The later
[original/candidate checkpoint](../review-checkpoints/2026-10-13/layernorm-original-and-candidate-real/receipt.json)
adds the candidate's independent real contract, and
[finite flat IO](../review-checkpoints/2026-10-13/layernorm-finite-flat-io/receipt.json)
passed 40 safety, output and frame gates for both complete programs. Valid row,
active allocation extents, separation, coverage and zero undef are explicit;
trace safety is proved from these conditions. Typed success and broader
source-profile admission remain separate gates.
The [uniform source and native checkpoint](../review-checkpoints/2026-10-13/uniform-source-and-native/receipt.json)
records 13 strict source tests, all six original storage ASTs, 24 selected-profile
source gates and 24 offline native compilations with 96 rechecked artifact hashes.
The native affine expression uses fused fp32 arithmetic; this is not an admission
of an unfused numerical model.
The [whole uniform real execution](../review-checkpoints/2026-10-13/uniform-whole-real-execution/receipt.json)
connects all 23 actual source statements to the independent four-stream store
formula, with successful execution and the original signed64 seed/offset domain.
It is distinct from the typed backend witness and independent replay.
After the aggregate, a checked bf16 conditional-cast counterexample exposed a
wrong register read on the branch that skips the cast. The
[constexpr continuation fix](../review-checkpoints/2026-10-13/conditional-cast-fix/receipt.json)
passed seven new axiom checks and 21 existing Lean regressions, preserving
path-specific carriers and fp64/fp32 arithmetic precision. Its
[pre-fix failure witness](../review-checkpoints/2026-10-13/conditional-cast-reproducer/receipt.json)
is retained. Subsequent frontend additions and aggregate revalidation remain
separate checks; the earlier aggregate receipt does not cover this change.
The [builtin zero checkpoint](../review-checkpoints/2026-10-13/builtin-float-zero/receipt.json)
adds exact `float(0)`, and the
[register dtype/where checkpoint](../review-checkpoints/2026-10-13/register-dtype-and-where/receipt.json)
checks six floating cast targets, mixed branch promotion and fp64 precision
preservation, with all 15 affected regressions passing.
The [backward softmax real checkpoint](../review-checkpoints/2026-10-13/backward-softmax-real/receipt.json)
proves the complete legacy real source's formula and frame. Its typed-source
erasure bridge and typed success family are separate work.
The [original backward native matrix](../review-checkpoints/2026-10-13/backward-softmax-native/receipt.json)
records 628 compiled and 500 rejected configurations out of 1128 offline probes,
with all 2512 successful IR artifact hashes rechecked. In particular, homogeneous
bf16 LOG branches fail the native exp type check and both homogeneous fp8 profiles
are rejected. Abstract Lean backend support does not override these native failures.
The later [softmax source-profile checkpoint](../review-checkpoints/2026-10-13/softmax-all-source-profiles/receipt.json)
checks all 36 complete original floating ASTs, 108 rejected expression
mutations and all 24 production source gates on frozen inputs. It does not
override the recorded native rejection of the two fp64-to-fp8 pairs.
The official independent comparator is unavailable in this container because
there is no systemd user manager; its sandbox requirements have not been
bypassed. No task has been added to the historical acceptance ledger here.

## Historical source inventory snapshot

The earlier integrated 2026-10-13 statement-scope snapshot records 175 Python/Lean port
pairs and 674 headline (`specification`) declarations in 278 specification files.
Of the imported ports, 103 use separate source/correctness/FP entries. Another
9 of the 184 kernel work directories remain source-import scaffolds and are not
counted as implemented ports, even when staged proof helpers exist. Layout and declaration counts are not completion
counts.
The `_spec-sheets` metadata directory is not a kernel work directory.

[`coverage_review.json`](./coverage_review.json) records the scope, assumptions,
numeric model, and Python function links for each headline. Its Lean and Python
source fingerprints are checked by
[`bench/coverage_review.py`](../coverage_review.py); the reviewed categories
are reflected in [`proof_gap_manifest.tsv`](./proof_gap_manifest.tsv).

| Coverage level | Headlines | What the label records |
|---|---:|---|
| `full_value_candidate` | 257 | A candidate full-value contract within its stated model and assumptions. |
| `specialization` | 382 | A contract restricted to the shapes, branches, parameters, or execution scope stated in the review. |
| `projection_only` | 4 | A formal projection whose extra casts do not establish native Python source coverage. |
| `precomputed_input_slice` | 26 | A contract whose inputs include prepared intermediates; it does not prove the omitted producer. |
| `pre_rounding_slice` | 3 | A real-valued result before the faithful fixed-width rounding/cast result. |
| `blocked_summary` | 2 | An explicitly blocked summary. |
| **Total** | **674** | **Across 278 specification files in 175 ports.** |

These categories describe the actual headline statements. A full producer
surface or projection elsewhere in a file does not automatically strengthen a
headline that consumes prepared values. A `full_value_candidate` label is not
an independent certification of the Python translation or GPU execution.
Axiom cleanliness and successful comparator replay are separate checks.
Historical issue links identify blocker families; they do not assert that an
issue is still open or that closing it completed the proof.

See [`proof_blockers.md`](./proof_blockers.md) for the remaining scope and frame
limitations, and the
[browsable coverage table](https://lizn-zn.github.io/VeriTile/proofs/coverage/)
for the individual theorem statements and source evidence.

## Earlier integration history — 2026-10-13

The merged Pow/source-audit and persistent-matmul public bundle is applied.
Its isolated candidate passed 188 regression tests, the complete source-only
audit and coverage/manifest checks, with 3075 unchanged input files. The new
public real-erased contract includes successful finite-flat execution,
independent GEMM outputs and a standard tagged-cell frame. Pointer-dtype source
selection is bound to the complete original and recorded native signature/TTIR.
Shared-tree revalidation and aggregate compilation remain separate gates;
source-gate success does not establish independent replay or full-port completion.

The random primitive and frontend are integrated after the complete 232-module
affected closure passed. The final continuation compiled 61 modules and reused
171 exact-hash artifacts; all 274 source files, dependencies and artifacts were
checked independently. Scalar-operation qualification repairs the import-name
collisions without changing arithmetic or contracts. Predecessor failures and
existing linter warnings are preserved in the receipts.

Seeded dropout is now imported with 19 checked modules, 82 standard-axiom
outputs and empty FP assumptions. Its two real headlines retain successful
finite-flat IO as a conjunct alongside standard tagged-cell realization and
the complete frame. The zero unspecified-load premise remains explicit in the
flat bridge. The FP theorem includes failures; a separate checked B=N=1
witness proves nonempty runs from empty registers for all 24 profiles in an
explicit consistency algebra, without claiming IEEE or all-shape typed success. All 24 storage/seed profiles compile offline at
the original B=1024, with no GPU execution or verified compiler refinement.
The strict identity-materialization source auditor is integrated too; its
isolated candidate passed 206 tests and all 24 production source-view gates.
No new translation exemption was added. The actual merged shared tree now
passes all 206 regression tests, the complete `--source-only` audit and the
coverage/manifest checks against the same unchanged 4158-file input snapshot.
All 175 selected views extract; the 43 existing translation exclusions remain.
Receipts are under `/tmp/veritile-shared-175-validation-20261013/`. Aggregate
Lean compilation, all-profile source coverage and official independent replay
remain separate pending gates.

An earlier completed shared aggregate exited 1 after 4747 Lake jobs, with
2971 captured source/build inputs unchanged. Its only listed failed module is
`bgmv_expand_slice.Proofs.FP`: the existing fp32 assumptions do not entail its
arbitrary-output claim. The public theorem is unchanged. The receipt is
`/tmp/veritile-shared-175-aggregate-20261013/receipt.json`, completed at
2026-10-13 13:35:34 UTC. It includes the random, seeded-dropout and persistent
helper integrations, plus the KL-triton public entries. The subsequent
`PhiloxRandom`, `PhiloxRandOp` and `RandomFrontend` regressions all passed on
the same unchanged inputs. This aggregate is a failure, not an acceptance.

Persistent matmul's reviewed ten integer-domain helpers and five hint helpers
are now integrated, with their two axiom-audit modules. They have 76 and 43
checked outputs respectively; independent fresh local replay reproduces all
five hint artifacts and their axiom log without diagnostics. The hint criterion
and reached tail counterexamples are scoped to a handwritten source-mapped
projection. Public source and theorem statements are unchanged. Neither
package proves full typed-source/TTIR refinement, compiler soundness or GPU
correctness. The updated source-only audit, coverage and manifest checks passed
against 4179 unchanged files; receipts are in
`/tmp/veritile-shared-175-native-hints-validation-20261013/`. The aggregate
result above includes these helpers. The isolated full `layer_norm_fwd`,
`multinomial_sampling` and `uniform_sampling` ports remain in progress.

The reviewed vector-seed and discrete-reduction packages are now integrated:
same-shaped vector RNG seeds, modular uint32 boolean sums and signed integer
argmax with leftmost ties, including the evaluators, memory lemmas and frontend.
The shared source-only audit passed after these integrations, as did 50 targeted
Python regressions and the coverage/manifest checks. That source run did not
capture a frozen-input snapshot; it is not proof replay. Its log and integration
hashes are under `/tmp/veritile-completion-20261013/`.

Direct Philox's complete isolated 224-module affected closure finished
successfully at 2026-10-13 14:00:16 UTC. Its receipt, 178 standard-axiom outputs,
sources, dependencies, artifacts and logs passed 30,222 hash checks (1,039
distinct file/hash pairs). The exact reviewed 18-file AST/evaluator/memory and
test package is now integrated, after matching every baseline and frozen local
dependency to the shared sources. No compiled artifacts were copied. These
historical build receipts do not certify the integrated tree or a complete port.
The source frontend remains staged and requires its actual execution tests;
the integer-cast candidate still has 40 successful compilations, one Attention
contract failure and 36 validated reuses. No successful full uniform port or
cast closure is claimed.

The post-Direct shared source-only audit and coverage check passed with all
4188 captured input files and the file set unchanged. The source audit includes
the proof-gap manifest gate; all 175 source views extract. It took 292.398
seconds, followed by the 17.589-second coverage check. The receipt is
`/tmp/veritile-completion-20261013/direct-shared-source/receipt.json`.
This checkpoint preceded only the documentation of these results; it contains
no Lean compilation or official comparator replay.

Current shared Lean compilation is unverified: sandboxed Lean cannot locate
its installation, and the required build's escalation was not executed because
automatic approval review failed with an account-policy error. This is a review
service failure, not a code-safety determination. It does not change the failed
BGMV aggregate or waive official comparator replay. No new complete-port count
is asserted; the historical 59-complete/125-pending record still needs a full
184-directory recheck.

## Earlier validation snapshots

The following records describe their respective frozen inputs. Their pending
statements do not override the current integration status above.

The earlier integrated coverage registry and proof-gap manifest passed for
the then-current 173-port snapshot. The shared-library and recent KL/RMS/GEMM/Rotary public-entry refresh
passes 3821 build jobs. FlashAttention, both RMSNorm implementations and floating/int32 Pow
have separate final public build and axiom receipts in `verification_round.md`.
A further aggregate refresh has finished with exit code 1 after reaching
4727/4727 jobs. Its only listed failed module is
`bgmv_expand_slice.Proofs.FP`, with 48 diagnostics. Its arbitrary-output theorem is not
entailed by the fp32-only rules, as the existing checked general countermodel
shows. The supported fp32 theorem is a separate scoped result. The aggregate
refresh cannot be recorded as a pass; earlier receipts do not certify it.
The latest 142 source/layout/coverage/sharding/dtype/constexpr/runtime/helper unit tests pass. Comparator-backed gate
tests cannot pass without the official comparator prerequisites; their failed
environmental run is not accepted proof replay.

The full `--source-only` audit is not yet green. Its initial run found a
whole-proof `trivial` support example (now fixed and Lean-checked), stale
translation-exception documentation, missing source-target metadata and
unresolved macro/profile source comparisons. The parser now follows local
source imports, honors explicit source selectors, expands supported quoted
splices and rejects ambiguous or unresolved views. It does not infer missing
parameter aliases or constexpr profiles; genuine source differences and
unresolved comparisons remain failures. Removing the five stale exception-list entries does not exempt those
ports or discharge their source obligations.

Official independent comparator replay remains unavailable because its
prerequisites and the systemd user runtime are absent. No new complete-port
count is asserted from these local results.

The later RMS/Pow source-metadata public refresh passes 3515 jobs. The earlier
source-only scan exited 1: 32 source views remained unresolved, ten ports lacked an
explicit JIT target, and the Flash/LlamaFF/Softmax source comparisons still need
reviewed mappings. Its transient manifest mismatch is separately resolved by
the final 668-record coverage and manifest checks. This remains an incomplete
corpus audit, with no additional textual exemptions.

The subsequent review adds explicit source-view comments to 37 ports and
refreshes 74 support fingerprints after checking that no Lean code or public
statement changed. Coverage and the proof-gap manifest still pass at 668
declarations across 274 public files. The frozen parser passes 81 regressions
and 14 independent adversarial checks. All 173 selected views extract and the
Python target-name gate passes; 130 enter detailed comparison and 43 retain
existing translation markers. All 43 markers existed before these changes;
the previous prose count of 42 was inaccurate.

That earlier full source-only audit exits 1, with stable fingerprints for
all 2439 audited inputs. Fourteen comparison gates fail across 23 ports.
Remaining obligations concern parameter aliases, dtype/cast and allocation
representations, arange and memory calls, constexpr profiles, helper expansion,
updates and control/statement order. The selected Flash actual-source and
SoftmaxOptimize dtype-attribute comparisons no longer fail. Unselected JITs and
profiles remain outside these textual checks; resolving extraction is not
all-profile certification. Its gate-to-port receipt is
`/tmp/veritile-source-audit-selected-final-receipt.json` and the full log is
`/tmp/veritile-source-audit-selected-v2.log`. A preceding run whose parser
changed during execution was discarded.

The next frozen full audit includes 45 exact parameter pairs in nine ports,
with comment-only Lean changes and 18 refreshed support fingerprints. The
corresponding tools pass 100 regressions and 14 independent adversarial checks.
The audit exits 1 with 14 comparison gates failing across 17 ports; all 2978
inputs, including per-port READMEs, remain stable. Extraction, inventory and
translation-marker counts are unchanged. That earlier frozen receipt is
`/tmp/veritile-source-audit-params-final-receipt.json`, with full log
`/tmp/veritile-source-audit-params.log`.

Post-snapshot fixes balance complete multiline loads before resolving dtype
attributes and accept self-negations only for explicit known floating storage.
The five context-attention ports and the selected fp32 RoPE view now pass
their production source-view gates, as does `kldiv_triton` after nine reviewed
parameter aliases. The two metadata edits change no Lean tokens and refresh
four support fingerprints. All 104 integrated regression tests and the
668-record coverage/manifest checks pass at that checkpoint; no remaining
profile/helper/scalar-cast obligation is waived by these targeted results.

The preceding integrated snapshot adds explicit boolean constexpr profiles and
57 reviewed parameter pairs for KL-ops, LlamaFF and both RMS-GEMM ports.
The four comment-only edits refresh eight support fingerprints. The parser
validates all original branches and the exact Lean selector/argument map
before specialization. All 114 combined tool tests and 21 independent
pointer/dtype probes pass; the 668-record coverage and manifest checks pass.
The frozen full source audit exits 1 with all 3001 input hashes stable.
Eleven comparison gates fail across seven ports: `kldiv_ops`, `l2_norm_bwd`,
`l2_norm_triton1`, `l2_norm_triton2`, `rms_norm_triton`, `pow_scalar_tensor` and
`softmax_flaggems`. Extraction remains 173/173; 130 views enter detailed
comparison and 43 retain existing translation markers, with no new exemptions.
That frozen receipt is
`/tmp/veritile-source-audit-profiles-final-receipt.json`, with full log
`/tmp/veritile-source-audit-profiles.log`. Neither this failed source audit nor
the separately failed aggregate build certifies complete-port acceptance.

The latest frozen source audit includes five recorded Python-float/fp32 runtime
ABI profiles and exact pure-helper expansion for FlagGems, which retains
symbolic storage. The six comment-only source edits refresh twelve support
fingerprints. Profile checks pin original source, full Lean arguments and
native TTIR; helper checks preserve arithmetic order and validate all original
branches before selection. All 142 integrated tests pass, alongside 36
independent helper cases, four helper production expectations and eighteen
runtime production expectations. Coverage/manifest counts remain unchanged.

This full audit exits 1 in 291.614 seconds with all 3019 inputs stable. Its ten
failing comparison gates affect only `pow_scalar_tensor`, down from seven
ports at the preceding checkpoint. All 173 views extract; 130 enter detailed
comparison and 43 existing translation markers remain. No new exemptions were
added. The exact receipt is
`/tmp/veritile-source-audit-runtime-helpers-final-receipt.json`, with full log
`/tmp/veritile-source-audit-runtime-helpers.log`. Recorded ABI/source views do
not certify all runtime values, native numerical behavior or complete ports.

After this snapshot, the pre-existing `softmax_triton3` marker is moved off its
block-comment opener and added to the completion list. This aligns the
documentation matcher's 42 entries with the detailed gates' existing 43 skips;
it creates no exemption and changes no Lean executable tokens. Two support
hashes are refreshed. Documentation/status, coverage and manifest checks pass.

The reviewed Pow candidate now passes the complete source-only audit in a
frozen worktree copy: exit 0 in 243.257 seconds, with all 3038 input files and
the file set unchanged. All 163 regressions and coverage/manifest checks pass.
The candidate retains all twelve original parameters, three casts, both
original branches, the helper/import bindings and memory fields; it binds the
selected fp32 profile to the reviewed native record. All 173 views extract,
130 receive detailed comparison and the same 43 existing translation exclusions
remain. The receipt is `/tmp/veritile-pow-root-final-receipt.json`. At that checkpoint the patches, including CI/docs registration and exactly
two support-hash updates, were ready but held outside the running aggregate. This candidate
pass is not acceptance of the current shared tree, unselected native profiles,
the existing translation exclusions or complete port contracts.

The shared fp32 decoder now supports exact signed-zero/subnormal projection,
with universal software-decoder compatibility and unchanged nonzero-exponent
results. The corresponding DSL validation and Core/DSL/bridge build pass 3306
jobs. The subsequent full aggregate rebuild is running; its receipt/log prefix
is `/tmp/veritile-fp32-finite-decoder-aggregate`. This local extension does not
complete the random primitive, seeded-dropout contracts or independent replay.
The actual scalar/vector DSL regression passes, including arbitrary vector
lengths and retained NaN/Inf rejection, and is registered in artifact CI.
Six bridge/vector axiom checks use only the standard axioms; the integrated
receipt is `/tmp/veritile-fp32-finite-decoder-integrated-receipt.json`.

The staged random conversion now has an all-word proof connecting the actual
two fp32 rounding stages to its dyadic formula and range `[0, 1 - 2^-24]`.
Seed/offset/width/round-count generalization, compatibility with the integrated
decoder and totality of the independent seeded-dropout formula also compile.
Fourteen modules are rebuilt from a frozen directory; their imports, nine
repository dependencies, all artifact hashes and 32 standard-axiom outputs
are checked. The receipt is
`/tmp/veritile-rand-prep/canonical-closure-20261013T082809629980Z/receipt.json`;
root review is `/tmp/veritile-random-canonical-root-review.json`. These new
modules remain in staging. Neither this proof nor formula totality supplies
public `tl.rand` support, complete seeded-dropout contracts or replay evidence.

The subsequent staged `Op.rand` core/memory integration passes its 44-module
dependency check and both numerical regressions, with 33 standard-axiom outputs.
Its source-profile checks preserve signedness, width and both canonical fp32
rounds; a reachable threshold example exercises the resulting dropout-mask
difference. The receipt is
`/tmp/veritile-rand-integration-overlay/core-checkpoint-receipt.json`.
A separate source-index bridge proves all i32 intermediates, actual BitVec32
multiply/add and signed-mask correspondence for symbolic legal block sizes,
including masked tail lanes. Nine standard-axiom outputs and all artifacts are
checked in `/tmp/veritile-seeded-source-domain-root-review.json`. Native evidence
still covers B=32 only. Shared integration remains pending; these staging checks
do not complete a corpus port.

The subsequent frontend/dependency staging build passes 112 selected modules,
including the new DSL tests and the existing finite-decoder/Powi/signed-load/
Triton smoke regressions. RandomFrontend prints eight standard-axiom checks
without diagnostics; dynamic Powi retains one unused-simp warning. The receipt
is `/tmp/veritile-rand-integration-overlay/frontend-checkpoint-receipt.json`.
A separate frozen build of all 232 affected library/test modules is running
from 274 captured Lean sources. Seeded dropout has six checked source-domain,
index, canonical-formula and writeback helpers, with 37 standard-axiom outputs;
root artifact review is `/tmp/veritile-seeded-dropout-helper-root-review.json`.
The writeback helpers establish raw region/finite-flat frames, but are not yet
a proof of complete source execution.

Persistent matmul's earlier checkpoint has 34 checked Lean helpers, including
source/FP, schedule, input-only GEMM formulas, complete concrete four-iteration
typed executions with raw memory frames, and generic K=0 memory preservation.
The concrete results cover both storage profiles and variants, with twelve
standard-axiom checks. An earlier source snapshot includes 38 helpers, adding
symbolic prefix execution, source-loop partition, masked-dot and fused-dot
partial-sum proofs. Four further modules checked after the freeze establish
signed loop control, grouped schedule injectivity/coverage and independent
masked input values: 42 Lean helpers in 45 source files, with fourteen more
standard-axiom checks. Five subsequent modules bring the set to 47 helpers in
50 files, completing the actual load sequence, masked store/reset with local
raw frame and signed iteration transitions. Twenty-eight symbolic axiom
checks pass at that checkpoint. Five more checked modules bring the set to
52 Lean helpers in 55 source files, covering actual group/coordinate/offset
setup and composed load/dot/last-K/store/reset execution. The composed step
proves independent GEMM readbacks at last K and a full raw-cell frame outside
active stores. Twelve further axiom checks bring the symbolic total to forty.
Latest execution stages use the refreshed Core/DSL, while older receipts
retain their original dependency scope. The receipt is
`/tmp/veritile-persistent-symbolic-stage-receipt.json`.
The latest ten-module addition brings the set to 62 Lean helpers in 65 source
files and completes the general positive-K original real execution, independent
GEMM readbacks for all assigned active outputs, A/B input preservation and the
full raw-cell frame. `RealContract.executes` starts from arbitrary registers
and assumes only the stated shape/stride, PID, region-separation and original
input-memory conditions. All 69 symbolic axiom checks use standard axioms;
root verified the new receipt/logs and every source hash in
`/tmp/veritile-persistent-full-loop-root-review.json`. Actual address safety is
checked, but whole-source trace safety, finite-flat contracts, candidate real
transport and public integration remain unfinished. It remains one of the
11 unimported directories. Later helpers are outside the frozen source-audit
snapshot. Neither this progress nor local native compilation supplies
independent replay or GPU execution evidence.

The next six checked modules complete persistent matmul's original whole-source
trace safety and both variants' flattenability. The shared staged set contains
68 Lean helpers in 71 source files, with 81 symbolic standard-axiom checks.
Root verifies all source hashes and the twelve new safety checks in
`/tmp/veritile-persistent-safety-root-review.json`. Finite-flat IO, candidate
real transport, public integration and independent replay remain pending.

The later persistent-matmul checkpoint contains 76 checked Lean helpers in 79
source files, with 116 symbolic standard-axiom checks plus twelve baseline
checks. Both variants now have successful full real-erased finite-flat
execution, independent GEMM outputs, input preservation and complete raw-cell
frames. Host-grid coverage, disjoint PID output windows, K=0 full-memory
preservation and a discharged finite nonempty witness also compile.
The standard `RealizesFrame_without_Rounding` wrapper uses `MemCell`, the exact
active output mask and an independently proved frame complement; it retains
the successful finite-flat contract as a separate conjunct. Three public
candidate declarations compile with standard axioms, and their two generated
spec sheets have no self-reference or missing summary. Root review is
`/tmp/veritile-persistent-standard-surface-root-review.json`.
Public/registry integration, native integer and legal-configuration admission,
dtype-branch source comparison and independent replay remain unfinished.
The staged named-surface scanner fix passes six regressions, preserves all
173 existing port classifications and recognizes the new candidate's actual
standard declaration without a custom-contract exemption. It is not applied
to the shared tools yet. The combined candidate suite passes 169 tests with
all 3039 captured input hashes stable. Inventory and complete-port counts are
unchanged.

## Audit Gates and CI Evidence

[`bench/audit_tritonbench_g.sh`](../audit_tritonbench_g.sh) enforces the gates
below. This table describes what is checked; it does not certify that an
in-progress run or a later revision has passed.

| Gate | Check and interpretation |
|---|---|
| Port inventory | Match Python/Lean file counts; reject compiled-port READMEs that still advertise TODO status. Scaffolds remain outside the implemented port set. |
| Port elaboration and proof replay | The final `bench/audit_trust.sh` gate elaborates the standalone corpus with the Lean trust/statement checks and requires official comparator export/replay. The aggregate audit also comparator-checks the proven library manifest targets. A separate duplicate compilation pass is not run. |
| Faithful-translation patterns | Check dtype-load additions, `keep_dims`, missing `+=` updates and normalized pointer lhs, `rsqrt`, Lean-only `tl.where`, `tl.*(...)` call set/order, control-flow counts, statement lhs order, and cast/scope notes. These scans do not establish arbitrary arithmetic equivalence. |
| Translation exceptions | Require explicit `Translation-surface blocker:` preamble markers and matching documentation. The 43 registered ports are listed below; a documented exception is not a discharged proof obligation. |
| Correctness surfaces | Require a named correctness surface or an explicitly registered `Correctness-surface blocker:`. The 10 registered ports and their 16 frame-free headlines are documented in `proof_blockers.md`; passing this gate does not put every headline on a named surface. |
| Statement-scope inventory | Check every headline against the manifest and the explicit review, including Lean/Python fingerprints and Python function links. |
| Placeholder proofs | Scan comment-stripped Lean for `sorry`, `admit`, `True := by`, and whole-proof `trivial`; elaborated trust checks and comparator replay provide additional proof checks. |
| Explicit algorithm blockers | Reject `hAlg` blockers. Their absence does not discharge specialization, prepared-input, rounding, frame, or translation limitations. |

For a standalone port check, [`bench/check_ports.sh`](../check_ports.sh) also
requires compilation and official comparator replay. `lake build` alone does
not cover the standalone corpus or all of these gates.

The website's
[recorded completed corpus audit](../../site/src/lib/coverage-ci.json), when
present, pins the successful run URL, commit, and completion time. A `null`
record means no completed public-repository audit has been recorded yet.
Check that commit and its
workflow when using the record as release evidence: an earlier success does
not establish that later changes passed newly added gates. Only a completed,
successful full run can be recorded with
[`scripts/record_coverage_ci.py`](../../scripts/record_coverage_ci.py).
Queued, running, cancelled, or skipped runs are not successful audit evidence.
The [Bench audit workflow](https://github.com/Lizn-zn/VeriTile/actions/workflows/bench-audit.yml)
provides the current run status and logs.

## Remaining Blockers

No explicit TritonBench-G `hAlg` blocker remains. Forty-three ports carry an
explicit `Translation-surface blocker:` preamble marker — a documented,
deliberate deviation of the Lean `triton { }` surface from a literal
transcription of the upstream Python body (helper-JIT inlining, constexpr-path
specialization, antiquoted in-body constants); see `proof_blockers.md` for the
per-port descriptions. `bench/audit_tritonbench_g.sh` keys the textual
py↔lean surface-scan exemptions on that marker only and requires this list to
match the active marker set exactly.

The current documented blocker set is:

- `bgmv_expand_slice` — dtype and cast specialization is staged by a source
  macro, the implicit output conversion is explicit, and early return becomes
  a sentinel guard. Full-source correspondence and precision/configuration
  regressions accompany the real IO and admitted-atom equivalence proofs.
- `attention_score` — fp16 Q/K/Out and fp32 M storage are explicit; the
  source's output element-type alias resolves to `tl.float16` for the supplied
  dtype configuration. Explicit region lists and the terminal cast are
  checked with dot/load precision and signed-window regressions.
- `attention_kernel` — fp16/bf16 storage specializations, region types and
  implicit bias-scalar conversions are explicit. The source regression checks
  storage bias precision, default dot outputs, casts and both B0 windows.
- `attention_fwd_triton1` — the six fp32 dtype-alias casts are spelled
  `tl.float32` to retain FP operations; `p_h.dtype.element_ty`, `b_q.dtype` and
  `p_o.dtype.element_ty` all select fp32 in the supplied tests. The README
  records the reviewed mapping; the lexical gate cannot establish it.
- `attention_forward_triton` — `_attn_fwd_inner` is fully inlined; fp16 Q/K/V/Out
  and fp32 scale storage are explicit, 128/96 head constants are symbolic, and
  the output element-type alias resolves to `tl.float16`. The source loop,
  typed dots, masks and store are reviewed and checked in Lean; the generic
  outer-call text scan does not align the inlined representation.
- `parallel_attention` — `parallel_rebased_fwd_kernel` + the
  `_parallel_rebased_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the backward shell and
  `_parallel_rebased_bwd_dq` are the trusted boundary.
- `bmm_optimized` — `bmm_kernel` now retains all eight constexpr mask arms,
  both CTA-reorder branches, the shortened final group, actual runtime grid
  dimensions and the three batch-pointer updates. The macro specializes
  optional masks, splits tuple assignment and makes range start/step explicit.
  Common fp32/fp16/bf16 storage and `allow_tf32=false` remain; implicit store
  conversion is explicit. Source masked loads still omit `other`, and the real
  product contract explicitly needs zero unspecified values. See the checked
  counterexample in `Proofs/UndefinedTail.lean`. Final replay status is tracked
  in `verification_round.md`; autotuning and grid composition are separate.
- `parallel_retention_attention` — `parallel_retention_fwd_kernel` + the
  `_parallel_retention_bwd_dkv` helper ported; the helper's scalar arguments
  become universally-quantified binders (no cross-JIT call surface; marker
  registered in `proof_blockers.md`); its trailing bare `return` dropped;
  its descending `-BTS` loop as the ascending `hi − j·BTS` change of
  variable with a `cdiv` trip count; the diagonal unary-minus decay index
  respelled as a subtraction under the causal `tl.where` mask; int→float
  promotions spelled `tl.toReal(...)`; the backward shell and
  `_parallel_retention_bwd_dq` are the trusted boundary.
- `cache_transform_triton` — both complete gathers specialize the nine independent
  cosine/sine fp32/fp16/bf16 storage pairs because region dtype annotations are
  syntax-level. Explicit signatures retain both inputs, signed metadata and
  both outputs. Original masks, omitted `other`, the full decoding tile and
  all strides remain; the only candidate edit commutes an exact index product.
- `triton_matmul` — `matmul_kernel` ported with both constexpr epilogue
  arms as twin surfaces (fp16 + fp8e4nv; marker registered in
  `proof_blockers.md`), one od-generic proof stack instantiated per arm
  (exec closed form + `⊨[R]` streaming face each); `tl.cdiv` trip count as
  the antiquoted `numKBlocks`; K-loop counter spelled `kk`;
  `launch_metadata` hook and per-dtype host config table are the trusted
  boundary.
- `attention_llama` — full `_fwd_kernel` sources for fp16 and int8/E5M2
  K/V, retaining both causal branches and source masks. The textual-scan
  exception covers constexpr storage specialization, explicit dtypes/casts
  and positional zero-fill written as `other=`. No FP8 arm is omitted.
  Correct.lean proves both versions' independent natural-exp formulas and
  complete flat IO, including original byte-cell inputs/frame, under positive
  block/count, whole streamed span, nonempty causal visibility, output
  injectivity and FP8 buffer-separation conditions. FPEquiv.lean proves the
  complete carry-first rewrite with add_commute and finite actual operands in
  every reached iteration. Both keep arbitrary source strides. Partial-tail
  conventional-attention formulas, empty visible-row formulas, host launch
  composition and hardware execution remain outside the real contract.
- `int4_matmul` — `matmul_kernel` ported at the `SPLIT_K = 1` store arm
  (markers registered in `proof_blockers.md`); GPTQ int4-dequantize GEMM
  proven as one exec closed form mirroring the `matmul_dequantize_int4`
  twin (packed `Region .nat` channels, per-lane group rows); the
  sub-first dequant runs on the `.int` channel via explicit `tl.cast`
  and promotes through `Op.intToReal` (first consumer of the
  signed-promotion lever); `tl.cdiv` trip count as the antiquoted
  `numKBlocks`; autotune sweep and host launch are the trusted boundary.
- `int8_dequant_matmul` — `_int8_matmul_rowwise_dequantize` ported at the
  `SPLIT_K = 1` store arm and the `EVEN_K = True` load arm (markers
  registered in `proof_blockers.md`); int8×int8→int32 GEMM with rowwise
  dequant epilogue, the first `Op.dotInt` consumer — the accumulator
  stays on the ℤ channel through the whole K loop and promotes to ℝ once
  via `Op.intToReal` in the epilogue; `has_bias` a genuine `Bool` with
  both arms under one headline; the faithful `.to(tl.float16)` casts make
  the masked store `.fp16`-typed, read back at the MemCell level; the
  autotune sweep and host launch are the trusted boundary.
- `int_scaled_matmul` — both JIT kernels launched and modeled in py
  order (markers registered in `proof_blockers.md`): the block-pointer
  int8 GEMM (first `.int`-channel block-ptr port — element dtype
  inherited from the typed base region via the port's inference rider,
  private `.int` boundary-check load lemmas, `tl.advance` walk,
  `MemCell.of .int` readback; step-form K loop spelled directly under
  `0 < BK` only) and
  the pointer GEMM with the inductor store suffix (descending loop as
  ascending substitution, `EVEN_K` a genuine `Bool` with both arms,
  `tl.broadcast_to` — whose syntax rider landed with this port — flat
  `col + N·row` store of the `Op.intToReal`-promoted `acc · s1(row)`
  read back as exact-`.real` cells); the host launches are the trusted
  boundary.
- `int8_matmul_kernel` — `matmul_kernel` ported in full (markers
  registered in `proof_blockers.md`); a pure-integer 2-bit-packed-weight
  GEMM proven as one fully dimension-general exec closed form — the
  first **nested-loop** `Op.dotInt` consumer (static field loop ×
  dynamic K loop; the A pointer advances continuously across both, the
  B pointer rebinds per field); `tl.static_assert` spelled faithfully
  (second consumer) and its divisibility is the headline's only K
  hypothesis; `.int`-typed store cells read back over launch-state
  accessors; the autotune sweep and host launch are the trusted
  boundary.
- `int8_matmul_quantization` — both JIT kernels launched and modeled in
  py order (markers registered in `proof_blockers.md`): the per-row
  float→int8 quantizer (two sequential K loops — a 0-seeded streaming
  abs-max pass, then the `Op.castRealToInt8` quantize-and-store pass;
  one exec headline with two MemCell readback conjuncts, int8 cells +
  fp16 scale cells) and the int8×int8→int32 GEMM (second `Op.dotInt`
  consumer, remainder-masked K loads at the honestly-ragged ceil-form
  `numKBlocks`, `SPLIT_K = 1` store arm, per-row × per-col scale
  epilogue promoted through `Op.intToReal`, fp16 MemCell readback);
  the autotune sweeps and host launches are the trusted boundary.
- `matmul_dequant_int4` — the target JIT is the file's second kernel,
  `dequantize_kernel`, the only one launched (the first, `matmul4_kernel`,
  is dead code byte-identical to the `matmul_dequantize_int4` port's
  kernel — not modeled; markers registered in `proof_blockers.md`);
  straight-line masked int4 dequantization proven as one fully
  dimension-general exec closed form whose only hypothesis is store-map
  injectivity; the signed nibble difference runs on the `.int` channel via
  explicit `tl.cast` and promotes through `Op.intToReal` (second consumer
  of the signed-promotion lever); the masked loads keep `other=0.0`
  faithfully; the autotune sweep and host launch are the trusted boundary.
- `matmul_dequantize` — all three JIT kernels launched and modeled in py
  order (markers registered in `proof_blockers.md`); three disjoint proof
  stacks mirroring the `matmul_dequantize_int4` / `int4_matmul` /
  `matmul_dequant_int4` twins; `matmul4_kernel` proven with `NO_GROUPS`
  as a genuine `Bool` parameter (both arms), `matmul_kernel` at the
  `SPLIT_K = 1` store arm with a MemCell-level `.fp16` readback (its
  source keeps explicit `.to(tl.float16)` casts), `dequantize_kernel` a
  1:1 mirror of the byte-identical 168th port; the three host launches
  are the trusted boundary.
- `attn_fwd_triton` — both helper JIT calls are inlined with independent
  pointer copies; the supplied bf16 Q/K/V/Out and fp32 scales are explicit.
  The value dot preserves its V-to-fp16 cast and fp16 result. Both defined
  outer stages and the finite additive penalty are retained. Head constants
  `128`/`96` become `BLOCK_DMODEL`/`HEAD_ACTIVE`; undefined helper stages are
  excluded. The changed historical model lives in `Proofs/LegacyExactMask`.
- `attn_fwd_causal` — both JIT helper calls are inlined with separate local
  pointers; fp16 Q/K/V/Out and fp32 scales are explicit. The `128`/`96`
  constants are generalized to `BLOCK_DMODEL`/`HEAD_ACTIVE`. Both defined
  outer stages and the finite additive penalty are retained; stages leaving
  the Python helper bounds undefined are excluded. The changed historical
  exact-mask model is isolated under `Proofs/LegacyExactMask.lean`.
- `attention_fwd_triton2` — `_attn_fwd_inner` inlined; `128`/`96` generalized.
  fp16 Q/K/V and fp32 scale storage are explicit; the output element-type alias
  resolves to `tl.bfloat16`. The explicit V-to-fp16 cast, fp16 dot result and
  bf16 terminal cast/store are retained and checked in the source regression.
- `attention_fwd_triton3` — the helper JIT is inlined and fp16 Q/K/V/Out,
  fp32 M/L and the source-dependent probability/output casts are explicit.
  The current shared source retains signed windows and bounds, KV tail checks,
  initial pointer advance, NaN alpha repair and all stores. Both real IO
  contracts cover independent strides, INIT/END, grouped heads and checked
  partial KV tails. The FP candidate changes only END normalization using
  div_mul_rcp under actual finite/nonzero intermediate-value guards.
  `Proofs/LegacySource.lean` preserves the altered historical transcription;
  its claims are excluded from the public headline inventory. The checked
  source-mask counterexample remains in `Proofs/SourceReview.lean`.
- `sgmv_expand_slice` — `EVEN_K`/`ADD_INPUTS=false`/`CAST_TYPE=false` path;
  CTA linearization via program-id axes; layout hints erased.
- `matmul_kernel` — in-body `4096` constants and `tl.cdiv` trip count as
  antiquoted binders.
- `matmul_triton_autotune` — `tl.cdiv` trip count as antiquoted `numKBlocks`;
  loop counter spelled `kk`.
- `bmm_chunk_bwd` — host-selected A/Dout storage and dot dtypes are
  specialized explicitly. The complete source now computes its own chunk
  length and retains all residual arms, partial tails and output casts;
  the former precomputed-length specialization lives in `Proofs/Legacy.lean`.
- `bmm_chunk_fwd` — host-selected A/B storage and dot dtypes are specialized
  explicitly. The nested causal early return is written as its complementary
  active-tile guard; skipped tiles perform no loads/stores. The actual
  in-body length, partial K blocks, signed sequence filter, original load
  storage and output conversion remain in the complete source.
- `iv_dependent_matmul` — only the `type == "pre_load"` mode mechanized.
- `matmul_tma` — `OUTPUT_F16` branch split into two surfaces.
- `softmax_triton3` — typed fp16/bf16 output stores make the implicit
  destination conversion explicit because pointer aliases lose their storage
  annotation in the current DSL. Source regressions check the carrier and
  single cast for every storage pair, candidate and mask branch; fp32 stores
  are unchanged.
- `relu_strided_buffer` — `one_tile_per_cta` branch split into two surfaces;
  `relu_forward` helper inlined; rank-1 stride-order constexprs (= 0)
  instantiated in `boundary_check`/`order`.
- `fused_recurrent_delta` — ternary pointer increments spelled as
  `IS_HEADWISE_BETA` constexpr if/else gates (one statement per arm);
  `do` → `do_`; `_` loop vars → `_i`; scalar-beta `… + T - 1` pointer inits
  ℕ-parenthesized.
- `triton_linear_activation` — `K_LOAD_MASK_NEEDED=True` heuristics arm
  specialized (descending K-loop as antiquoted ascending trip count);
  activation helper JITs inlined; module constants antiquoted; unused
  `CACHE_KEY_*`/`SPLIT_K` dropped; `ACTIVATION` as a `String` parameter.
- `rbe_triton_transform` — `get_freq_multi_tokens` helper inlined; helper
  `DIM = 128` → `DIM : Nat` binder; dtype/shape-changing `freqs` rebinds
  renamed; `.to(tl.float32)` spelled `tl.toReal`; explicit int→float
  promotion bindings.
- `chunked_cumsum_fwd` — bias storage and finite/positive-infinite upper
  bounds select explicit typed source templates. Quoted clamp/multiply
  statements retain the complete preprocessing, masks, stores and scan.
  The default infinite minimum remains opaque in FP semantics. The public
  real contract states the host-valid remaining-length condition; both full
  real and FP paths are checked. Independent acceptance is tracked in
  `verification_round.md`.
- `chunk_retention_ops` — all four typed sources retain signed `tl.arange`
  through an explicit `.to(tl.int32)` identity cast. The backward-state loop
  uses NT-1-j, preserving per-chunk stores, scaled queries, weighted incoming
  gradients and full-BT carry decay. All tile dimensions remain independent;
  the empty loop is valid. Complete typed equalities connect the other three
  source pairs to the shared proofs in `chunk_retention`.
- `chunk_retention` — all four complete typed kernels and their candidates
  retain decay intrinsics, casts, masks and stores. Signed `tl.arange` is
  explicit as `.to(tl.int32)`, adding an identity cast to the call sequence.
  Backward-state's descending loop is an ascending change of variable;
  `NT > 0` is required for its original post-loop-index use. Its square dot
  shapes, unused inputs, dead history load, final-only decay and sole final
  store are preserved. The real contracts expose zero unspecified values,
  access bounds and a full tagged-cell frame; see the case README.
- `block_sparse_attn` — complete typed source specialized for fp32/fp16/bf16
  storage, preserving signed CSR metadata and all mask/channel branches.
  Q-storage probability casts and implicit output conversions are explicit;
  `static_print` has a fixed debug payload. The candidate commutes only score
  scaling. Correct.lean exposes independent-array real flat IO; FPEquiv.lean
  exposes the admitted rewrite. The legacy natural-metadata summary is kept
  in Proofs/LegacySummary.lean and is not a current public contract.
- `bgmv_shrink_kernel` — `-1` sentinel early return as a guard (write-free
  path proven); `SPLIT_K == 1` store-vs-atomic tail split into two surfaces;
  `tl.max_contiguous` hint erased; unused `xk_stride` dropped.
- `layer_norm_welfold` — `tl.full` positional dtype spelled `dtype=`;
  `tl.broadcast_to` via tuple `tl.broadcast`; fused `tl.sum(…,1)[:, None]`
  split into two statements; unmasked `tl.store`; `libdevice.rsqrt` spelled
  `tl.rsqrt`; register casts parenthesized.
- `fused_layernorm_triton` — inductor `welford_reduce`/`welford` helpers
  inlined as exact-ℝ closed forms; `tl.broadcast_to` via tuple
  `tl.broadcast`; unmasked `tl.store`; `libdevice.rsqrt` spelled
  `tl.rsqrt`; register casts parenthesized.
- `fast_ce_loss` — DO_SOFTCAPPING and DO_LOGIT_SCALING are staged through
  cap/scale before dtype inference. All four combinations retain complete
  source templates; full statement-tree comparison and FPFastCeSource check
  branch/cast correspondence. The old pointer-load/cast exception is removed:
  the signed label pointer bump and direct load cast now remain literal.

The translation-marker list above is separate from the per-headline proof-scope
inventory. The manifest has 302 rows outside `full_value_candidate`: 268
specializations, 29 prepared-input slices, 3 pre-rounding slices, and 2 blocked
summaries. The blocked summaries are in `quant_transpose_kernel` and
`bgmv_expand_slice`. The former's companion scaled-store theorem and the
`quantize_global` and `rowwise_quantization_triton` headlines are pre-rounding
slices. None of those slices proves the faithful CUDA `llrint` / final int8 result. The per-statement review and
`proof_blockers.md` retain the remaining limitations.


## attention_kernel migration

The public directory now has Kernels/Correct/FPEquiv entries for both fp16 and
bf16 storage. Both originals and reciprocal candidates have independent real
IO proofs: scaled Q·K and both B0 bias windows feed the base-2 weighted sum,
with successful flat execution, active-address safety and a memory frame.
The real scope is contiguous head rows, unit inner stride, positive block/head
sizes, P_SEQ=0 and one or more whole key blocks; head and bias strides are
symbolic. Arbitrary-layout and partial-key-block real correctness remain outside
this contract. No identity-fp16-rounding premise is used by the current entries.

The FP proof uses only div_mul_rcp at actual post-loop fp32 operands, under
finite accumulator and finite/nonzero denominator checks. It retains every
source stride, both bias windows, storage-dtype bias arithmetic/scalar conversion,
default fp32 dots, Q/P casts, exp2, loads and store. Nonempty 2×2 full-source
executions for both dtypes inhabit the domain in a model of the admitted rules.
The textual translation exception now concerns explicit storage specializations
and implicit scalar/dtype conversions, checked by the source regression.
Historical cast-bearing claims are retained in Proofs/Legacy.lean; the new real
proofs execute a checked cast-free source and do not import that legacy model.
Final replay status is recorded in verification_round.md.


## attention_score: signed-source replacement

The current Kernels/Correct/FPEquiv entries retain signed offsets and sizes,
all four flags, fp16 Q/K/Out, fp32 M/accumulation, both checked-load arms and
masked output stores. Both real projections implement an independent finite
Q/K/M column sum, with complete trace safety and flat IO. Positive common
query/key block size is the only dimension premise; partial query blocks and
empty loops are included. The IO windows require unmasked M padding, K reads
before empty loops, and output allocation only for active keys.

The scalar-left FP candidate uses only mul_commute at finite actual fp32
operands in every reached iteration. Typed-source regressions retain the
precision boundaries and IS_EVEN_N-controlled query-tail selection.
Independent comparator replay and trust checks accept all 33 case/regression
files; all 198 named proofs are in the actual inventories, and snapshot inputs
match the final sources. This is the fifteenth completed directory.

The historical natural-subtraction window is preserved only in
Proofs/LegacyNaturalWindow.lean. Query 0/key 1 is included by that model but
excluded by the source's ordinary positive signed window. Its old statements
are absent from the public scope inventory. attention_kernel_aligned remains
user-deferred and is not a prerequisite for this work.


## attn_fwd_causal: complete two-helper source contracts

Kernels.lean retains both defined outer stages, independent helper pointer
copies, source strides, masks and finite additive penalty. Correct.lean now
proves both complete erased kernels against an independent normalized base-2
weighted sum. The formula retains repeated visits and masked-load padding;
it does not silently replace the finite penalty with exact causal masking.

The real contracts include successful flat-memory execution, all actual
address bounds and preservation of every cell outside active output writes.
Only actual input reads are pinned. Scale loads remain unmasked even for
fully padded key tiles. The partial, unaligned-range regression checks this
allocation requirement. No loop invariant or output equation is assumed by
the public contracts. Shapes and independent source strides are symbolic.

FPEquiv.lean proves complete-program division-to-reciprocal equivalence using
only div_mul_rcp. Guards inspect actual fp32 accumulators and denominators
after both loops; all source casts, dot precisions, masks, stores and failures
are retained. TypedExecution supplies a consistent-model execution/domain
witness. Experimental dimensions do not restrict the symbolic derivation.

RealFlatCounterexample supplies a concrete disjoint flat allocation and
successful execution of both versions: Q/K=0, scales=1 and V=[0,1] yield a
strictly positive first output. Thus the real source is not exact causal
masking. This is not a claim about GPU fp16 underflow. The earlier changed
exact-mask model remains in Proofs/LegacyExactMask and has no current public
coverage rows. Host launches and fixed-width address overflow are outside
these per-program contracts.

Final validation: `TritonBenchGVerified` builds (3,723 jobs); both focused
source/IO regressions, all 29 metadata/layout/shard tests and all source gates
pass. Independent comparator/trust replay accepts all 48 case/regression
files, including 307 named proofs and 1231 kernel theorem targets;
all 824 snapshot input hashes match. Evidence: `Logs/comparator-check-sve03cl2`.

## attn_fwd_triton: bf16 two-helper source contracts

Kernels.lean contains both complete sources with bf16 Q/K/V/Out, fp32
scales/accumulators, a bf16-input fp32-output score dot and explicit fp16
probability/V casts before the fp16-output value dot. The candidate changes
only final division to reciprocal multiplication. Both legal outer stages,
finite additive penalty, repeated visits, pointer reset and masks remain.

Correct.lean displays an independent weighted-sum formula and both complete
real flat IO contracts, including actual input windows, successful execution
and the memory frame. RealProjection checks equality of the complete erased
sources with the verified two-helper causal source before transferring these
contracts. It does not assert FP equivalence between the two benchmark cases.
The real exact-mask refutation also applies to actual flat source execution.

FPEquiv.lean retains the bf16 source and uses only div_mul_rcp, checking actual
finite fp32 accumulators and finite/nonzero row denominators after both loops.
Its consistent-model execution witness covers both defined outer stages.
Host launch composition and fixed-width overflow remain outside the per-program
scope. The historical changed model has no current public coverage rows.

Final validation: `TritonBenchGVerified` builds (3,733 jobs). The bf16 source
regression, refreshed causal source regression, 29 metadata/layout/shard tests
and all source gates pass. Independent comparator/trust replay accepts all
12 case/regression files, including 176 named proofs and
567 kernel theorem targets; all 834 snapshot inputs match.
Evidence: `Logs/comparator-check-_zp921x6`.

## batched_vecmat_mult: complete streamed GEMV contracts

Kernels.lean contains both complete fp32 sources. The candidate swaps only the
two operands of each vecmat addition; the dynamic K loop, input broadcasting,
axis-2 reduction, transpose and unmasked store are unchanged.

Correct.lean exposes both independent real double-sum formulas and complete
flat-memory IO contracts. Dimensions remain symbolic, with K=BK*T, BK>0 and
BN≤N plus bounds for the displayed unmasked input/output windows and disjoint
allocations. The source does not use M; actual access windows determine
validity. Both contracts include successful execution and every non-output
cell's preservation, including T=0. RealSource proves whole-source erasure
correspondence and the candidate's complete flat execution equality.
StreamReal adds the exact real two-stream/two-program-ID IO face; identity
rounding is confined to its internal bridge to the existing real foundation.

FPEquiv.lean uses only add_commute, checking finite actual carry and block-sum
operands at each reached update after the shared input prefix. Complete state
and failure are preserved. Source regression checks fp32 loads, the leading
unit broadcast axis, reduction/transpose expression, loop bounds and store.
RealExecution witnesses disjoint flat memory at nonzero program IDs, a 2×3
output tile and two K blocks: all six output values are 16. TypedExecution
witnesses both complete executions and satisfiable domains in a consistent
abstract model. Neither witness is a GPU experiment. Host launch composition,
fixed-width address overflow and concrete IEEE identities remain outside the
per-program contracts.

Final validation: `TritonBenchGVerified` builds (3,743 jobs); the precision/geometry
regression, 29 inventory/layout/shard tests and every source gate pass.
Independent comparator/trust replay accepts all 11 case/shared/regression
files, including 80 named proofs and 472 kernel theorem targets;
all 844 final snapshot input hashes match.
Evidence: `Logs/comparator-check-m5wnbzx6`.


## FP8 conversion source mapping

`f8_conversion_utils` now uses explicit typed region declarations and the
source auditor's checked implicit-destination-cast correspondence. It no
longer needs a translation-surface exception. Both complete JITs retain all
input/output precisions, load defaults, masks and the duplicate forward store.
The current real contracts erase precision for the copy formula, while typed
FP equivalence preserves every conversion. Host reinterpret/launches and
physical rounding remain outside these per-program statements. See its
case README and verification round for the current proof acceptance status.


## fast_ce_loss source migration

The three current typed sources preserve the pointer bump, direct signed load
cast, libdevice tanh, constexpr flags, precision transitions and all original
stores. Their complete statement trees match the three Python JITs after only
Lean notation and explicit implicit-destination cast normalization. The old pointer-load/cast exemption is removed; the remaining constexpr
specialization is documented and all four flag combinations are checked. Six real specifications expose the actual
padded LSE/partial-loss/backward formulas and finite flat IO; three full FP
index commutations have an empty domain and no numerical atoms. The nonzero
chunk write and vocabulary-only capped-tail claims have source counterexamples.
Integrated builds, affected regressions and independent replay are tracked in
verification_round.md; migration itself is not acceptance.


## Fast normalization and rotary source migration

fast_layernorm, fast_rms_layernorm and fast_rope_embedding now each expose
complete Kernels/Correct/FPEquiv entries, independent full finite-flat real
contracts and empty-domain structural FP equivalences. These retain the
padded LayerNorm variance, RMSNorm's source-specific weight strides and cast,
and the complete rotary head loop with initial coefficient snapshots.
Storage/cast checks and public finite/typed execution witnesses are included.
The source auditor accepts whitespace before tl.store parentheses through its
existing argument parser, with negative pointer/value/type regressions.
Builds and independent replay are tracked in verification_round.md; source
migration and scope classification alone do not mark the cases accepted.
