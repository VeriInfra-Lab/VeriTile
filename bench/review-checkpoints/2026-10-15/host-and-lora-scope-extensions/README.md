# Logical host and LoRA scope extensions

This batch adds thirteen Lean modules and two public execution contracts: the complete per-program LoRA
branch family for eighteen selected storage profiles, and logical allocation
plus the original single-program fp16 matrix-transpose wrapper. Whole-program
acceptance remains **0 of 183**. These additions do not close the remaining
caller domains, native allocation/addressing, compilation or concurrency gaps.

[catalog-supplement.json](catalog-supplement.json) is cumulative. Its first eight
rows are unchanged from
[the preceding batch](../original-scope-extensions/catalog-supplement.json):
three source-binding-only rows and five tasks with eleven public contracts.
The two new rows add one public contract each, bringing the cumulative
extensions to thirteen public contracts across seven tasks.
[root-review.json](root-review.json) binds only the newly reviewed source and
scope metadata; the prior rows retain their prior review references.

| Task | New contract and limits |
|---|---|
| `lora_expand_gemv` | Six input/weight pairs × three output formats; every EVEN_K, ADD_INPUTS and CAST_TYPE choice; signed sentinel; complete output loop; independent per-block memory recurrence and full frame. Output/index dtype completeness, host/grid, native signed addresses and concurrent alias behavior remain open. |
| `matrix_transpose` | Arbitrary fp16 payloads in distinct logical regions; logical allocation and one original kernel; all transposed output bits, input preservation and full frame. Empty-shape strides use `max(dim,1)`. Native error paths, actual allocation, returned Tensor metadata and ABI/device correspondence remain open. |

The LoRA reference freezes the input vector and index at program entry, then
reads each later weight/output block from its current memory. Its fp32
`CAST_TYPE=False` reduction uses an explicitly supplied software reduction
tree. No used numerical callback falls back to an arbitrary backend. Native
rounding, conversion, reduction order and fusion still need validation. The
source dtype inspection used the recorded Triton **3.7.1** package.

The matrix wrapper always requests fp16 storage. Quantifying over arbitrary
sampled fp16 words avoids an RNG distribution assumption; the transpose does
no floating arithmetic. Its logical model accepts natural dimensions that the
native `tl.arange` frontend may reject, including zero or non-power-of-two
endpoints. Those missing source/error cases are structural work, not numerical
deferrals. Shape and stride helper definitions are metadata; the public theorem
contains a shape-product equality, not a returned native Tensor theorem.

[numerical-followup.json](numerical-followup.json) preserves the earlier three
follow-ups and adds the precise LoRA software/native correspondence work.
These entries are not registered acceptance `scope_bindings` in the unified
numerical queue. Matrix transpose introduces no new numerical assumption.

The existing [metadata baseline](../original-scope-extensions/metadata-baseline/receipt.json)
is reused unchanged. It contains the 893 pre-extension public claims; no new
baseline or filtered historical scan is created for this batch. The cumulative
supplement yields 906 public claims when the current catalog is reproduced.

[evidence-index.json](evidence-index.json) maps the retained compact receipts,
logs and CPU wrapper probe to their producer hashes. Original receipts retain
their historical `/tmp` paths. The LoRA producer's original `theorem` declaration
is accompanied by a separate reviewed revision to the public `specification`
keyword and a successful recompile; the old receipt is not rewritten. The
matrix wrapper probe substitutes CPU allocations and intercepts the launch;
it does not execute a GPU kernel.

Individual proof compilations and independent reviews are retained under
[evidence](evidence). The complete registered Lean build passed **7,244 jobs**,
and the updated workflow passed **25 sharding tests**. The final
[validation record](validation.json) binds the exact inputs and observed exits.
Source, baseline, catalog, value-metadata and deferred-numerical checks passed.
The full-scope gate still fails for 183 pending reviews, and the global runner's
comparator checks fail because its host lacks the official comparator.
The isolated library pass for preceding commit `93d38e39` does not include this
batch. No current-corpus independent replay success is inferred. The preceding
210 benchmark-tool test result is historical; those tests were not rerun here.

Use a fresh output directory when reproducing the global checks:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/host-and-lora-scope-extensions/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/host-and-lora-scope-extensions/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/host-and-lora-global-replay
```
