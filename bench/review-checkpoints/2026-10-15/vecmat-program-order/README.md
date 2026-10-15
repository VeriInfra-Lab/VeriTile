# Batched vecmat program order

This batch adds four Lean modules and one public execution contract. The
cumulative extensions contain nineteen public contracts across eight tasks.
**Whole-original-program acceptance remains 0/183; all 183 are pending.** Every
original stage, dtype and branch is still required. Only explicitly registered
numerical assumptions may be deferred; structural gaps remain open.

`batched_vecmat_arbitrary_program_order_correctness` executes the complete
original fp32 logical source for any permutation of the full host grid. Each
program receives fresh registers and the preceding actual memory. The final
memory equals both the independent initial-input reference and the existing
`gridExpectedMemory`, with the complete untouched-memory frame.

The explicit condition is `A ≠ output ∧ B ≠ output`; A and B may alias each
other. Exact body load registers and offsets establish the read footprint.
Complete execution preserves input regions, and actual grid/lane addresses
establish disjoint writes between valid programs. Exact scatter readback
connects stored words to rounded accumulated values. These results reduce
actual program effects to fixed initial-input overlays, whose disjoint writes
commute. No floating reassociation law or different-tree equality is used.

Internal partial schedules may be empty or repeat valid program IDs. The public
permutation runs each valid ID exactly once. Out-of-grid IDs are excluded
because this source has unmasked stores. Empty grids and K=0 are included;
parameter-level nondivisible cases retain the existing natural floor grid.
Original wrapper guards remain covered by the separate host contract.

This closes arbitrary sequential order of complete logical programs under the
separated profile. Native allocation freshness, physical byte aliases, pointer
and integer widths, caller/error domains, specialization/resources, source
compiler correspondence and instruction/warp interleavings remain structural
obligations. The same `FP.Config`, valid reduction plans, leaf/seed ordering and
block accumulation are retained. The five existing numerical follow-up rules
are copied unchanged; no new callback, numerical law or deferral is introduced.

[root-review.json](root-review.json) binds the independent source-level review,
all four new sources, project dependencies and exact public statement. Producer
receipts and logs are preserved byte-for-byte in [evidence-index.json](evidence-index.json).
Temporary `.olean` files, dependency symlinks and drafts are excluded. Previous
declarations and source views are unchanged; the new stage is
`separated_fp32_arbitrary_sequential_program_order`. The shared DSL source view
keeps `single_program_total_typed_memory`. The original metadata baseline is
reused unchanged.

The complete registered Lean build passed **7,271 jobs** with actual exit 0;
all **25 sharding tests** passed. Final global results and exact source/input
bindings are recorded in [validation.json](validation.json). Historical
`93d38e39` comparator evidence excludes these new modules. The separate replay
being prepared for `51518a8c` also excludes this batch and has no result claimed
here. Earlier benchmark-tool tests are reused only after checking that their
tool/test inputs are unchanged.

The global runner exited 1: scope review correctly reports all 183 original
programs pending, and the host's official comparator is unavailable to its
preflight/library commands. Source tools, structure, baseline, catalog
reproduction, legacy-source comparison, value metadata, deferred numerical
metadata and unchanged-input gates passed. These failures remain explicit;
the batch does not claim a green whole-corpus acceptance result.

Reproduce the global checks in a fresh output directory:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/vecmat-program-order/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/vecmat-program-order/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/vecmat-program-order-global-replay
```
