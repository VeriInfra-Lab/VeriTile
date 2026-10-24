# int_scaled_matmul

The current canonical source is
[`revision001/int_scaled_matmul_repaired.py`](revision001/int_scaled_matmul_repaired.py),
SHA256 `17bd051b849369e11f3c2b57440052a8c304722b32985499300ab572c553a33b`.
The selected Python repair is unchanged. Both the unscaled and row-scaled
APIs are covered. Start with [`Kernels.lean`](Kernels.lean),
[`Correct.lean`](Correct.lean), [`FPEquiv.lean`](FPEquiv.lean), and the exact
shared [`Contract.lean`](Contract.lean). Source/model citations and the native
operand-order binding are in [`SOURCE_BINDING.md`](SOURCE_BINDING.md).

## Independent result and admitted domain

For A of shape M×K and B of shape K×N, define

```
W[m,n] = signed32(sum(k=0..K-1, signed8(A[m,k]) * signed8(B[k,n])))
```

The unscaled output is W. The independent Real interpretation of the scaled source is
`signed32(truncTowardZero((W : ℝ) * rowScale))`, using ordinary Real multiplication
and Mathlib floor/ceiling. Integer projections and the signed output type stay
explicit. This mathematical interpretation is separate from native FP behavior.
The scaled FP source converts W to FP32 with nearest-even
rounding, multiplies by the FP32 row scale, then converts toward zero to INT32.
`KernelExecution.expected` reads the original logical Tensor values from the
common physical byte reader; it has no kernel execution or tile schedule.
This integer-format result includes modulo-2^32 overflow. It is not an
unbounded Real or FP32 matrix-product claim.

A/B are INT8, C is caller-supplied INT32, and optional row scales are M×1 FP32.
The unchanged assertions require rank-two contiguous strided CUDA tensors on
one device, resolved negative views, matching shapes, legal tile parameters,
the source's signed-index envelope, aligned representable accessed byte
windows, and C separated from read-only inputs. Read-only inputs may alias;
contiguous storage-offset views and irrelevant singleton strides are admitted.
The launch passes canonical dense strides. Empty outputs have no program or
pointer requirement; K=0 is covered. All INT8 values and all FP32 scale words
remain admitted. No finite-scale, defined-conversion, or no-overflow guard is
added. See the retained [exact contract description](revision001/README.md#checked-contract).

## Complete kernel proof

[`IO.lean`](IO.lean) displays the exact sources, finite typed A/B matrices,
row-scale vector, output matrix, strides, masks, full grouped-grid schedule and
empty scratch list. Input masks are false on empty output grids, so those
launches impose no input backing or pointer requirement. `Formula.dot` is the independent whole-K signed8 product
with signed32 wrap; `Formula.scaled` adds the retained FP32 suffix and total
conversion port. Read-only aliases remain allowed.

`Correct.int_matmul_correct` proves the independent signed8 sum with signed32
wrap. `Correct.int_scaled_matmul_real` connects the complete scaled source to the
independent ideal-Real scaling formula above. `RealScaled.truncate_spec` derives
its toward-zero inequalities from Mathlib. The Real interpreter uses exact
integer-to-Real promotion and ordinary multiplication, with a total mathematical
signed32 projection after truncation; it makes no assertion of native agreement
on undefined FP casts. Both Real results include actual source access safety,
successful full-grid execution, every logical output and the complete exterior
cell frame. All original integer-overflow, K-tail, group-tail, K=0, empty-grid,
strided-layout and read-only alias cases remain.

`Kernels.lean` now contains four full ordinary `triton` programs. Both originals
show explicit signed8 load conversions and signed32 accumulator updates.
`SourcePreservation.unscaled` and `.scaled` prove complete program equality with
the previously certified width projection by `rfl`; the frozen Python and
revision001 evidence remain unchanged. The candidates have a direct optimization
purpose: `unscaledCopyFree` removes `c = accumulator` and stores the accumulator;
`scaledSharedIndices` reuses the prelude's unchanged `rm` and `rn`, removing their
two repeated epilogue computations. No measured speedup is claimed.

`FPEquiv.int_matmul_pair` and `FPEquiv.int_scaled_matmul_pair` are the actual
whole-grid `original ≡[[]] candidate` specifications. Candidate execution and
safety proofs follow the real prelude, loop predecessor states, branches, load
masks and stores. The scaled loop also proves that the shared row/column vectors
remain unchanged. Both sides implement the same independent full output formula
and preserve the entire exterior memory. Scale-load lanes, masks, FP32 promotion,
multiplication order and total conversion port are identical; every FP32 word
and every local undefined-cast policy remains admitted. These pairs use no
experimental numerical rewrite. The empty rule set is relative to the selected
strict integer and FP32 software primitive model.

`Misc/NativeCorrect.lean` retains `Correct.int_scaled_matmul_correct` as the
ordinary all-word native formula theorem, together with
`FPEquiv.int_matmul_equiv` and `FPEquiv.int_scaled_matmul_equiv`. Those old names
remain auxiliary source/native bridges. The main theorem/Proofs dependency
closure does not import `Misc/`.
[`NativeIO.lean`](NativeIO.lean) selects fixed K32 saturating instructions and
separate modular carry updates on the same protected source. The scaled variant
also selects the inspected reversed FP32 multiplication operands.
`Proofs/NativePrimitive.lean` proves these primitive identities; both actual
signed8 load paths select the fixed instruction branch. No whole-dot numerical
law or finite-input restriction is assumed.

`Proofs/NativeComplete.lean` connects those concrete schedules to the original
`IntegerInstructions` and `CheckedNative` full-memory executors under the exact
original signed-index envelope, with its empty-grid exception. Former raw
statements live as ordinary theorems in `Proofs/RawCompatibility.lean`.
`Misc/TensorBridge.lean` retains Python admission, Tensor/physical word and byte
observations, pointer/storage-offset bindings and all original frame results.
The quoted source and all these transports reuse the same original complete
source and finite-instruction proof foundations.

The existing complete source, finite-index, grouped-grid, Tensor and byte
proof foundations remain retained. The direct programs are the current readable
source authority; proof-only decompositions stay in `Proofs/`. Historical public
entries and the frozen source/evidence tree remain unchanged.

## Integer instructions and floating profile

The current Python already uses a zero-seeded tile `tl.dot(a,b)` followed by an
explicit inline `add.u32`. The checked tile bounds imply BK≤65536; every
signed8 product has magnitude at most 16384. Thus every tile prefix stays
within signed32, and the retained `SaturatingTile` proof discharges saturation
for arbitrary instruction partitions. The carry add wraps. The historical
carry-fed saturating-MMA counterexample remains separately documented in
[`NATIVE_BINDING_CORRECTION.md`](NATIVE_BINDING_CORRECTION.md); it is not restored
or relabelled as a numerical experiment.

`KernelModel` fixes the shared concrete FP32 primitive profile: nearest-even
promotion/multiplication, no FTZ, canonical arithmetic NaNs, and toward-zero
signed conversion on its defined domain. An undefined conversion retains a
local `Word → BitVec 32` result. This makes the output word type explicit without
inventing a value or excluding a scale. The old three primitive-law interfaces
are proved from these definitions internally. They are not deferred numerical
experiments in the current kernel revision.

The inspected scaled PTX multiplies **row scale first**, converted accumulator
second. `KernelNative.multiply_comm` proves all-word equality in the selected
profile, including signed zero, infinity and canonical NaN. This result is
connected through `instructions_eq`, `run_eq` and `bytesRun_eq` to the complete
native and physical executors used by the public theorem. It has no finite
operand premise and is not an experimental rewrite rule.

The source-to-ISA evidence is the retained Triton 3.7.1 SM90 lowering: six current
profiles cover both kernels, both scaled EVEN_K branches, and 16×16/64×64 tiles
with BK=32, GROUP_M=8, four warps and two stages. MMA16 has a zero seed; WGMMA64
uses scale-d=0. Scalar sites show `cvt.rn.f32.s32`, non-FTZ `mul.f32`, and
`cvt.rzi.s32.f32`. This is an inspected compiler profile, not a universal compiler
or architecture proof. The selected canonical-NaN primitive model does not
claim native NaN-payload conformance or hardware results for undefined casts.

There are **no deferred numerical transformation assumptions** in this revision.
Historical generic-backend results and their old documentation are preserved;
their scalar-interface experiment wording is not the current experiment queue.

## Validation and limits

The focused checker executes the canonical AST with real CPU backing storage
and a mocked CUDA context: 61 prelaunch rejections and 50 interpreted calls
passed. Its sole inline-assembly adapter implements the exact audited
`add.u32`; loads, masks, dot and stores use Triton's interpreter. Cases include
K/group tails, K=0 and empty outputs, offset/singleton layouts, read-only aliases,
byte windows, the signed-index boundary and full-K integer overflow. This is
not GPU numerical validation.

The retained native validation receipts are under
`.veritile-cache/task-deliveries/int_scaled_matmul/`. The direct-source preservation,
new Real source theorem, actual two-program pair checks and fresh joint entry
check for this revision are recorded under
`.veritile-cache/finish-pending-105-20261021/int_scaled_matmul/`. Existing compiler artifacts
are checked and pinned without repeating their compilation. Independent review
is recorded separately. No GPU run, independent numerical replay or independent
cold build is claimed. Runtime grid-product overflow/segmented-launch support
remains outside this kernel-only work.

## Optional raw and Tensor interfaces

`RawCorrect.int_matmul_correct` and `RawCorrect.int_scaled_matmul_correct`
retain the original arbitrary-memory whole-grid formulas and frame.
`RawFP.int_matmul_equiv` and `RawFP.int_scaled_matmul_equiv` retain successful
finite-index/native execution and exact equality of the entire memory.
These ordinary theorems are in `Proofs/RawCompatibility.lean` and are exposed
through optional `TensorBridge.raw_*` aliases.

`TensorBridge.int_scaled_matmul_correct` and
`TensorBridge.int_scaled_matmul_equiv` retain both original Tensor application
statements, including physical word/byte observations and frames. The existing
Tensor adapter and shared IO reuse `BlockWord.grid_correct`,
`TotalScaled.grid_correct` and the same finite-instruction refinements;
`KernelExecution.reference_eq` preserves the explicit logical-input binding.
Historical `revision001` sources and accepted numerical boundaries are unchanged.
The format revision performs no new GPU experiment or independent replay.

The current two-program comparison uses the complete `unscaledCopyFree` and
`scaledSharedIndices` candidates described above. Their top-level statement
counts are 15 versus 16 and 23 versus 25, respectively;
`SourcePreservation.unscaled_changed` and `.scaled_changed` prove that both
candidates differ from their originals. The selected Python source, numerical
rules and input contract are unchanged.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.
