# Vecmat values and logical view addresses

This batch adds twelve Lean modules and two public execution contracts. The
cumulative extensions contain eighteen public contracts across eight tasks.
**Whole-original-program acceptance remains 0/183; all 183 are pending.** The
standard requires every original stage, dtype and branch. Only explicitly
recorded numerical assumptions may be deferred; structural gaps remain open.

| Task | Added result | Remaining boundary |
|---|---|---|
| `batched_vecmat_mult` | Complete original fp32 software-value kernel, every K/BK iteration, independent memory reference/frame; ordered logical allocations and guards, complete sequential 2D grid; output bounds at K=0 and unique program/lane write addresses | Natural-only host domain; native allocator/storage/ABI, integer/address widths, specialization/launch legality, input/output separation and concurrency; native numerical bindings |
| `kcache_copy_triton` | Universal forward rank4-to-rank3 successful-view address/read preservation for the unchanged logical stride-inference algorithm, including all sixteen symbolic singleton partitions and arbitrary natural strides | Native Torch/allocation, physical aliases and storage offsets, signed/native-width metadata, full caller/dtype/error domain, source stride/shape limitations and concurrency |

The two batched-vecmat public declarations name separate `full_kernel` and
`logical_wrapper_and_complete_sequential_2d_grid` stages. The original wrapper
has no dtype argument: all three allocations explicitly use `torch.float32`.
Thus fp32 covers its enumerated wrapper dtype domain, subject to the unbound
allocator/storage representation. `FP.Value .fp32` supplies concrete words in
the historical DSL `real` channel; that channel does not mean mathematical
real arithmetic. Products, the chosen valid reduction tree, explicit zero
seeds, transpose and ordered block accumulation retain their software rounding
and order. No real-sum, reassociation or fp16-rounding-is-identity law is used.

The `host_nat_allocator` profile makes three requests before the ordered K/M/N
guards. It preserves allocation failures and zero-divisor error order, and
executes the entire original grid with preceding actual/reference memory.
Allocation is an explicit non-numerical primitive boundary; native freshness,
zero initialization, contiguous storage and error heap/RNG effects are not
proved. The functional theorem handles every produced word without an RNG
distribution assumption. Twelve accepted original-wrapper CPU captures and
twelve error cases support metadata review; they do not execute GPU kernels or
cross-evaluate Lean. Four accepted captures lie outside the natural host domain.

The output helper has no K-loop-index premise. It proves bounds and injectivity
for the actual logical output addresses, including stores after K=0. These
results do not imply input/output separation, native race freedom or arbitrary
schedule-independent values. Coverage of every tensor output/K coordinate
still uses the original positive/divisible host conditions.

The kcache helpers derive stride coefficients from the actual chunk/consume
algorithm over sixteen symbolic singleton partitions; sizes above one remain
arbitrary. They derive the returned shape/strides/region from successful
`reshape` with `copied=false`, prove the flattened leading index is in bounds,
and equate reads with a common logical storage offset. Zero/broadcast strides
need no injectivity or contiguity premise. A zero dimension gives an empty
indexed domain, while returned stride-list length is still proved. This is
forward flattening preservation in the explicit Lean model, not native Torch
refinement. Nine explicit `axiomsClean` commands check recursive dependencies
of exported results. `auditModuleAxioms` selects headlines/legacy suffixes;
its presence alone does not enumerate every theorem.

[root-review.json](root-review.json) records three independent reviews and the
exact integration decisions. The immutable producer catalog proposal is kept:
the final batched row filters global library dependencies out of its task-local
hash map, keeps the shared DSL memory semantics at
`single_program_total_typed_memory`, and describes host composition separately.
Only the existing kcache host declaration's review and scope narrative change;
its statement, role, dtype and stage are preserved. Other prior declarations
and source views remain unchanged. Supporting address/view theorems add no
public contract.

[numerical-followup.json](numerical-followup.json) preserves all four earlier
rules and adds precise fp32 scalar-policy, reduction-tree/zero-seed and
fusion/contraction follow-up for batched vecmat. No new numerical law hypothesis
is introduced. These descriptive rules are not automatically accepted
`scope_bindings`; structural gaps cannot be deferred through them.

[evidence-index.json](evidence-index.json) retains exact producer receipts,
independent reviews, final compiler logs and CPU probes. Temporary `.olean`
files, dependency symlinks and drafts are excluded. Producer receipts preserve
their original temporary paths and pre-review status. The original 893-claim
[metadata baseline](../original-scope-extensions/metadata-baseline/receipt.json)
is reused unchanged. [catalog-supplement.json](catalog-supplement.json), the
generated catalog and the 183-task ledger bind the integrated source set.

The full registered Lean build passed **7,267 jobs** with actual exit 0, and all
**25 sharding tests** passed. Final global checks and exact input bindings are
recorded in [validation.json](validation.json). Structure, source tools, legacy
source comparison, baseline, catalog reproduction, value metadata, deferred
numerical metadata and unchanged-input checks passed. The global runner exited
1 because all 183 scopes remain pending and this host lacks the official
comparator for its preflight/library commands. The isolated official comparator
replay of `93d38e39` excludes this batch and provides no current-source or
whole-task acceptance. Earlier benchmark-tool tests need not be repeated when
their tool/test inputs are unchanged; their exact reuse is recorded separately.

Reproduce the global gates in a fresh output directory:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/vecmat-value-and-view-address/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/vecmat-value-and-view-address/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/vecmat-view-global-replay
```
