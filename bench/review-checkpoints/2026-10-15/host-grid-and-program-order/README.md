# Logical host grids and FP4 program order

This batch adds eleven Lean modules and three public execution contracts.
The cumulative extensions now contain sixteen public contracts across seven
tasks. **Whole-original-program acceptance remains 0/183; all 183 are pending.**
The user's standard requires every original stage, dtype and branch. Only
explicitly recorded numerical assumptions may be deferred; structural gaps
remain open work.

| Task | Added result | Remaining boundary |
|---|---|---|
| `kcache_copy_triton` | Ordered logical wrapper, both layouts, 3D/4D reshape, natural `n` including zero and multiple tokens, actual complete sequential 3D grid, independent current-memory reference and frame; every materialized reshape-copy value | Universal successful-view stride/address preservation; full caller/metadata dtype domain; native memory, addresses, launch and concurrency |
| `lora_expand_gemv` | Ordered logical wrapper and errors, singleton-4D squeeze, exact N/K/strides and host-derived flags; actual complete `(N,batches)` grid with prior-memory reads and independent reference/frame | Unchecked output/index dtype domain; native metadata, addresses, allocation, admission and concurrency |
| `fp4_to_bf16` | Any permutation of complete grid programs yields exactly the initial-byte output reference and full frame; helpers also cover partial, repeated and extra inactive IDs | Native instruction interleavings, physical aliases/allocator, unrestricted input dtype and host caller/error domain |

Each new contract uses a separately named stage in the cumulative
[supplement](catalog-supplement.json). Earlier public declarations and source
views are unchanged. The three updated rows bind the union of predecessor and
new sources through [this root review](root-review.json); other rows retain
their exact predecessor review references. Host composition does not change
the shared DSL's per-program memory semantics or assert verified Python
lowering. The LoRA producer's proposed broader source-view label is retained
as historical evidence, with the final registration choice documented in the
root review.

The kcache theorem has six inherited logical floating constructors and nine
discrete constructors. Its fp64 carrier remains opaque legacy storage; the
existing raw floating evidence covers five machine formats. Lengths and block
tables always use mathematical signed integer cells, without independent
dtype descriptors. The original ignores cache-last and lengths strides and
leaves legacy shape checks absent; those limitations remain explicit. The
36 original-wrapper CPU cases and 35 Lean metadata checks support the source
review but do not replace a universal view-preservation theorem.

LoRA retains all six asserted input/weight pairs and explicitly selects
fp16/bf16/fp32 output. The original has no output/index dtype assertion. Index
metadata does not select execution semantics, which remain mathematical signed
cells. The real wrapper defaults `add_inputs=True`. The modeled six-shift
`next_power_of_2` helper yields zero at K=0, causing rejection before launch;
N=0 yields an empty logical grid, whose native legality is unproved. The 576
accepted CPU metadata combinations and 14 ordered error cases do not execute
GPU kernels or cross-evaluate the Lean model.

FP4 input preservation and disjoint logical program writes are derived from
actual source execution. The public premises are the inherited packed uint8
input, distinct logical output region and permutation of the complete grid.
The result concerns arbitrary sequential orders of complete program executions;
GPU instruction/warp interleavings remain outside this theorem.

No new numerical law is introduced. The existing four follow-ups are preserved
in [numerical-followup.json](numerical-followup.json), including the native
rounding, conversion and reduction-plan bindings for LoRA. These descriptive
follow-ups do not themselves establish registered acceptance scope bindings.

[evidence-index.json](evidence-index.json) maps the exact retained producer
receipts, independent reviews, compiler logs and CPU probes. Producer receipts
keep their original temporary paths and pre-integration status. Temporary
`.olean` files and dependency symlinks are excluded. The original 893-claim
[metadata baseline](../original-scope-extensions/metadata-baseline/receipt.json)
is reused without regeneration.

The complete registered Lean build passed **7,255 jobs** with actual exit 0,
and all **25 sharding tests** passed. Catalog reproduction, source checks,
structure, value metadata, baseline, deferred numerical metadata and unchanged
input gates passed. [validation.json](validation.json) binds the exact inputs
and observed exits. The global runner exits 1 because all 183 scopes remain
pending and its host lacks the official comparator for preflight/library replay.
The isolated official comparator's historical replay of commit `93d38e39`
excludes this batch and cannot certify its current sources. Earlier benchmark
tool tests were not rerun; their relevant sources remain unchanged.

Reproduce the global checks in a fresh output directory:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/host-grid-and-program-order/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/host-grid-and-program-order/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/host-grid-order-global-replay
```
