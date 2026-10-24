# Vector addition floating grid and original scope inventory

This batch adds nine Lean modules and one registered public specification.
Cumulative extensions contain twenty public contracts across nine tasks.
**Whole-original-program acceptance remains 0/183; all 183 are pending.** All
original stages, dtypes and branches remain required. Only explicitly scoped
numerical correspondence may be deferred; structural gaps remain open.

`vector_addition_typed_grid_correctness` proves complete ascending logical-grid
execution, a result formula from initial input memory, and the full memory
frame for 25 ordered pairs of fp16/bf16/fp32/f8e4/f8e5 storage formats. Operand
promotion and conversion back to x's storage format are explicit. Every
program reads the preceding actual memory with fresh registers. Earlier flat
tile writes cannot affect later tile inputs, even when x/y/output have equal
logical region names. The positive-block premise permits the ceil-grid proof;
zero-length and partial final tiles are included.

The public theorem uses an arbitrary floating backend, including fp32 addition.
Distinct-format conversions are also arbitrary; same-format conversion is
defined as identity. Separate `SoftwareFP32` helpers instantiate concrete
`FP.add cfg` in both legacy and typed channels. Those helpers do not instantiate
the public arbitrary backend or prove native conformance. Neither finiteness
nor commutativity nor real-arithmetic equality is assumed.

`TypedFloat.source_projection` proves dtype/cast-erased body equality with the
unchanged original DSL source. The manually constructed typed AST is reviewed
separately; it is not labeled as additional shared-DSL syntax or as task-local
SC admission. The source view selects only `add_kernel` in `Kernels.lean` and
retains `single_program_total_typed_memory`. Native compiler translation remains
unproved. Masked loads without `other` retain unspecified inactive values;
active-output correctness does not assume those lanes are zero or finite.

Logical memory has dtype tags, and wrong-tag reads return defaults. Same-region
aliasing at identical natural element offsets does not establish shifted,
strided, mixed-width or otherwise overlapping physical byte views. Storage
metadata, allocation, input/output bounds, finite-width addresses and native
instruction interleavings remain structural gaps.

The retained dtype audit covers 17 canonical signatures and all 289 promotion
pairs: 217 promoted and 72 rejected by the inspected promotion rule. The Lean
table matches every captured observation, independently checked with explicit
native/Lean dtype-name reordering. Only 25 promoted pairs have typed execution
proofs in this batch; 192 remain. Thirteen original-AST frontend probes produced
eleven TTIRs and two errors on the recorded sm90 environment. This is neither
a complete native backend matrix nor GPU execution. fp64, fnuz, integer/bool,
unknown Torch dtype families and exact error domains remain explicit work.

`HostWrapper` preserves fallible, stateful `empty_like` before the CUDA guard,
output-derived numel, block1024/ceil grid, launch, return and error-state order.
The callback receives all tensor descriptors; no shape, equal-length, stride,
contiguity, dtype or same-device guard absent from the original is invented.
The harness models seed0, sequential x/y random draws and add for sizes98432,
1024,2048,4096, dictionary keys and module assignment. Its `logicalLaunch`
executes the legacy scalar source and ignores native dtype/layout metadata;
typed native host dispatch is not composed. Those helpers therefore do not
close the original host/harness stages in the public stage inventory.

[root-review.json](root-review.json) records the independent nine-module review
and exact source/metadata bindings. Eight frozen producer modules contain one
ordinary theorem. A separately reviewed ninth module exposes the same result
as a public specification through direct delegation, preserving all producer
bytes and existing catalog validation. Forty-three named axiom checks and eight
noncircularity checks passed. One harmless tactic-style warning in `Value.lean`
is retained. The producer's build-config freeze named `lakefile.lean`; the actual
`lakefile.toml` is separately bound by the independent and root reviews.

[numerical-followup.json](numerical-followup.json) preserves all five prior rules
and adds three narrowly scoped followups for native fp32 addition, arbitrary
floating add callbacks and distinct-format conversion routes. bf16/fp16 binding
must include the frontend's via-fp32 route where applicable. Native RNG
algorithm/state/default-dtype mapping remains structural host debt; no random
distribution law is required for logical kernel correctness on arbitrary words.
These descriptive followups are not accepted unified queue scope bindings.

[evidence-index.json](evidence-index.json) retains exact producer receipts,
compiler logs, dtype probes, eleven TTIRs and independent reviews. Temporary
`.olean` files, dependency symlinks and drafts are excluded. All prior public
declarations and source views are preserved, and the original metadata baseline
is reused unchanged.

The complete registered Lean build passed **7,280 jobs** with actual exit0;
all **25 sharding tests** passed. Final global checks and exact input bindings
are recorded in [validation.json](validation.json). Historical and separately
running `51518a8c` comparator replays exclude these nine modules. Earlier
benchmark-tool tests are reused only after confirming unchanged tool/test inputs.

The global audit completed with actual exit1. Source tools, structure, unchanged
metadata baseline, catalog reproduction, legacy source correspondence, value
metadata, deferred numerics and unchanged-input gates passed. Scope review failed
because all183 original-program scopes remain pending. Comparator preflight and
library replay failed because the official comparator is unavailable to the host
runner. These failures are retained; no current-batch comparator pass is claimed.

Reproduce global checks in a fresh output directory:

```sh
python3 -B bench/current_source_audit.py --root . \
  --catalog bench/review-checkpoints/2026-10-15/vector-addition-floating-grid/current-source-catalog.json \
  --baseline-package bench/review-checkpoints/2026-10-15/original-scope-extensions/metadata-baseline \
  --source-map bench/current_source_selections.json \
  --legacy-inventory bench/tritonbench_g/current_completion_inventory.json \
  --evidence-root bench/review-checkpoints \
  --review-manifest bench/review-checkpoints/2026-10-15/raw-copy-software-metadata/catalog-review-rebased.json \
  --supplement bench/review-checkpoints/2026-10-15/vector-addition-floating-grid/catalog-supplement.json \
  --scope-ledger bench/original_program_scope.json \
  --output /tmp/vector-addition-floating-grid-global-replay
```
