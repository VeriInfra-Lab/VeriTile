# SGMV full-backing byte separation

This is an independent contract/source candidate. The canonical wrapper and
its documented DLPack counterexample remain frozen. See
`../BYTE_SEPARATED_REPAIR.md` for the exact domain change.

`Memory` contains local copies of existing generic byte/store lemmas, avoiding
unrelated cached kernel imports with conflicting generated equation names.
It introduces no allocation, scheduling or separation premise.

`Contract` retains all original nonseparation checks and replaces backing-ID
inequality with nonoverlapping complete untyped-storage byte ranges.
`Guard.python_separate_iff` proves the exact executable overlap test.
`Binding` uses actual core FP32 X/Y Tensors and ordinary TensorLaws, together
with mixed-width W/metadata windows. Nonempty view containment, 64-bit live
storage bounds, full byte tails and zero-length pointers are explicit; no
separation or result premise is hidden in the binding. `logical_read` connects
the core Tensor's logical values to the raw words used by source execution.

`TensorBridge.checked_real` proves complete typed-source success and ordinary K-term
contractions folded in original segment order. `TensorBridge.checked_fp` and
`TensorBridge.checked_caller_bytes` retain the complete native typed-source execution,
every touched output byte, actual supplied Tensor object, None return, every
byte of all five original read backings, and the complete complement frame.
Store payloads are read from final source memory. Independent references do
not depend on source execution. The preserved source and generic complete
source proof chain are reused; no old identity-separation Binding is used.

`Bounds` derives signed-64 clamped offsets, `Access` derives actual 64-bit
pointer words and independently sized metadata spans. `Physical` also proves
zero-column/zero-batch/inactive cases and arbitrary partial-failure frames.
`OutputFrame` spells out untouched output cells and slice padding. A sentinel
segment may overlap a live one, so preservation is asserted only for cells
untouched by all active segments.

All add branches, all eight metadata width combinations, K=0 and K/M/N tails,
singleton 4D W, aligned offset views, read/read aliases and ordered overlapping
segments are retained. The local numerical boundary remains the historical
FP conversion/add and eight m16n8k16 ports with live FP32 carry plus two
zero-conversion laws. There is no finite-value guard, scalar-FMA replacement,
whole-result oracle or new GPU execution claim.

The historical `check_byte_separated.py` records candidate-only actual-source
CPU controls against the historical source pins. `PublicAudit` checks the public
import closure for axioms and circular target definitions. Root acceptance is
required before changing source selection or ledger counts.

Historical validation: 15 candidate modules compiled freshly, with a 222-module repository
import closure and public axiom/noncircular audits. The actual-source CPU
checker passed 33 complete calls, 57 original rejection cases, partial failure
frames, and 116 additional interval cases (80 DLPack overlap, 20 adjacency,
16 empty/partial/tail edge cases). All 90 old task files and 92 frozen defect
files were unchanged at that checkpoint. No new GPU execution or offline kernel compile ran.

`Correct.lean` now exposes `sgmvIO`, the independent ordinary contraction
formula, and one `specification` through the shared `ReadProgramIO.ImplementsOn`
judgment. `Misc/NativeCorrect.lean` exposes the same source jobs and IO with the existing
native instruction recurrence as an ordinary theorem. This native
result is a source observation, with the existing primitive ports and zero
conversion laws; it introduces no optimization or native-to-Real equality.

The public IO lists typed inputs, output, regions, original strides, masks,
metadata and the full ordered schedule. Only active segment cells are writable;
all logical output cells are read back, including unchanged cells. The shared
read-view judgment preserves the original decoded-read input domain and exact
exterior cell tags. The FP16 channel's low-16-bit encoding condition retains
the existing typed input-word representation and admits every FP16 pattern.
It is not a finite-operand assumption or an experiment to defer.

`Proofs/SourceAccess*.lean` connects declared bounds to the actual source
metadata branches, every signed reduction-loop iteration, masked in-place
loads and stores. `Proofs/IO*.lean` reuses the prior complete source results,
checks the complete launch schedule, and composes the exact frame. The previous
raw public theorem statements remain as ordinary theorems in
`Proofs/KernelRealExecution.lean` and `Proofs/KernelNativeExecution.lean`.
The complete caller-Tensor and byte observations remain in `Misc/TensorBridge.lean`.

All metadata widths, source scalars/flags, mixed precision, skipped segments,
K=0, tails, overlapping segments, read/read aliases and optional addition are
retained. The historical native and GPU evidence above keeps its original scope.

## Complete implementation pair (2026-10-22)

`FPEquiv.sgmv_grid_equiv` compares the actual original kernel with
`IndexReuse.kernel` from `Candidate.lean`, through the complete original
ordered read-view IO schedule:

```lean
original rest ops a p shape segments ≡[[]] indexReuse rest ops a p shape segments
```

The candidate reuses the tile offsets already computed before the K loop:

```python
offset_cm = cur_seq_start + offset_m
offset_cn = offset_n + slice_offset
```

The original recomputes the two `arange`/tile-vector expressions at this point.
Reusing them removes duplicate integer coordinate construction; no speedup is
claimed without measurement. The appended Python entries are
`_sgmv_expand_slice_kernel_index_reuse` and `_sgmv_expand_slice_index_reuse`.
The original complete Python module remains an exact prefix, and both wrappers
have identical checks, arguments and launches apart from the selected kernel.

`Proofs/IndexSource.lean` proves equality of the changed integer expressions and
their complete successor states. `Proofs/IndexKernel.lean` connects these states
to the actual complete source, derives loop preservation of the reused offsets,
and proves candidate source safety. `Proofs/IndexIO.lean` composes actual runs in
original segment order, including overlapping writes. The main conclusion
includes both source-safety traces, both successful complete executions, every
logical output read (including untouched cells), and both exterior frames.

The full selected FP32 X/Y and FP16 W/cast domain remains: all FP32 words,
every canonical FP16 word pattern, all eight independent signed32/64 metadata
ABIs, both addition branches, arbitrary original K and tile tails, skipped and
empty cases. Original shape, layout, separation, metadata and binding conditions
remain explicit. The dot, live FP32 carry, casts, loads, stores and FP operation
order are identical. The empty experimental rule set requires no floating
rewrite, zero-conversion assumption, new numerical experiment or deferral.

`Correct.kernel_real` retains the independent ordinary K-term contraction,
optional addition and ordered overlap specification. Single-source native
results remain ordinary auxiliary theorems in `Misc/NativeCorrect.lean`; their
historical primitive/zero-conversion laws are separate from this exact pair.
The old `ByteSeparated/Kernels.lean`, `source_pin.json` and numerical/compiler
evidence describe the historical byte-guard repair and retain their old hashes.
Current source preservation and candidate checks are recorded under
`.veritile-cache/finish-pending-105-20261021/sgmv_expand_slice/`.
The current `check_candidate.py` there checks the exact two-assignment change,
wrapper equality and Python syntax, runs 33 complete calls and 57 rejections
for the actual candidate, and reuses the retained parent and 116 interval
controls. Historical pin checks read saved exact sources; historical reports
are preserved. All checks and a fresh joint Real/main/native import passed.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.
