# First-NaN, first-index argmax repair

`triton_argmax_nan_first.py` changes only the three JIT bodies of the historical
`triton_argmax_repaired.py`. The wrapper, guards, allocations, requested scratch
dtype, dimension `contiguous()` call, return shapes, and launch segmentation
are unchanged. The 57 pinned original files remain untouched.

Every floating reduction explicitly selects the first NaN if one exists;
otherwise it selects the first numerical maximum. The source reduces separate
integer NaN flags and sanitized numerical values, then rereads the selected
original value. The dimension loop commits a new NaN only while its running
value is not NaN, so later tiles cannot replace the first NaN. Numerical ties,
both zero signs, and infinities retain their least logical index. The flags are
explicit int64 constants in Python and the shared quotation; they do not change
the requested input or scratch dtype.

The global target preserves the actual two-stage request: first-index FP32
selection in each chunk, requested scalar conversion and storage, decoded
scratch selection, then the original first-stage index gather. It is not always
the original FP32 input's argmax. For example, FP16 scratch on
`[1.0001] + [-9.0]*3 + [1.0002] + [-9.0]*11` returns 0, while the original-input
FP32 argmax is 4. The default FP32 case has a separate proof that its result is
the first NaN, or the first numerical maximum when no NaN exists.

The historical measured defect is the dimension update: `[1, NaN]` and
`[-3, NaN, 2]` returned 0 instead of 1. Saved global interpreter controls passed;
they are not reported as global counterexamples. Explicit global NaN handling
removes dependence on a reduction helper whose comparison has no explicit NaN
ordering.

## Proof path

All modules use namespace `VeriTile.Bench.Repairs.TritonArgmaxNanFirst`.

| Layer | Checked endpoint |
| --- | --- |
| Exact old admissions | `Contract.lean` aliases `Checked.InputContract` and executable checks |
| Input-only FP32 target | `reference_spec`, `first_nan`, `padding_lower` |
| Requested global target | `staged_maximum`, `global_reference`; target has no implementation tree |
| Readable shared source | `Kernels.lean`: floating specializations and the complete dimension loop |
| Shared discrete reductions | `TypedReduction.selected_eq`, `IntegerReduction.shared_tree` |
| Whole native dimension quotation | `Typed.source_reference`, `TypedNative.dimension` |
| Floating global quotations | `TypedScratch.stage1_run`, `stage2_run`; `TypedNative.first_float`, `second_float` |
| Integer global quotations | `TypedIntegerMacro.source_native`, `TypedNative.second_integer` |
| Live native grid and byte readback | `Native.global_source`, `dimension_source`; `Result.global_readback`, `dimension_readback` |
| Current byte memory to complete quoted launches | `BoundSource.first_step`, `second_step`, `global_run`, `dim_step`, `dimension_run` |
| Actual wrapper Tensor and caller frames | `TensorBridge.global_tensor`, `dimension_tensor` |
| Default FP32 stronger target | `Default.staged_reference`, `NativeKernelResult.default_readback` |
| Arbitrary-Real shared execution | `Correct.global`, `dimension`; prior raw endpoints in `Proofs/RealKernelResult.lean` |

The native quotation equations identify actual selections, original-address
rereads, scalar conversions, scratch stores, and current-memory gathers with
the live native graph. `BoundSource.entry` and `dimEntry` instantiate every
quotation's scalar loads from its actual current physical byte state.
`FirstStep` commits the requested cast of the selected original-address reread
and the quoted index to that same state. `SecondStep` uses the actual current
scratch and index bytes. `global_run` chains all first-stage commits, derives
floating scratch reachability from those physical stores, then executes the
second quotation and commits its gathered scalar. Dimension programs group
the existing jobs into their full 128-lane CTAs; `dimCommit` reads all active
output payloads from the quotation's final heap and encodes their byte stores.

These traces are proved conclusions. The public `TensorBridge.global_tensor`
requires `BoundSource.Binding` for the selected request: floating requests
carry `Rounding` and `FloatCodec`, and integer requests carry the existing
`ZeroClamp` scalar laws. `fp64_cast_order` derives scalar order preservation
from this codec, excluding an FP64 conversion that reverses two FP32 inputs.
The former unbound graph-only results remain explicitly named
`global_graph_tensor` and `dimension_graph_tensor` auxiliary lemmas.

The public Tensor theorems conclude `GlobalRun`/`DimensionRun` ending at the
same `result.state` as the independent index readback and caller-byte frame.
The byte graph composes these operations over the grid and proves readback and
byte frames. The Tensor endpoints retain the unchanged
wrapper's actual returned allocation/view, shape, int64 dtype, and caller bytes.
Routine positive/active geometry is derived from the existing admission.
Dimension dtype arguments remain ignored, including invalid tokens, exactly as
in the original wrapper. Empty outer/inner products return an empty Tensor and
have vacuous element readbacks; there is no new nonempty-output guard.

The Real theorem executes the entire first grid and second quotation. Its
scratch loads are derived from first-grid stores, and its target quantifies over
arbitrary logical Reals. The dimension theorem executes all range iterations
and final stores. Precision casts erase locally. For integer/bool scratch the
explicit idealization erases the entire NaN0/clamp/truncate/store/load macro;
`IO.projectedKind` uses the Real value channel for that erased macro.
This projection does not identify native results for every requested dtype with
the Real argmax, and does not claim that a finite Real clamp is identity.

## Local boundaries

Native FP32 equality, ordering, NaN classification, and reduction selection are
concrete bit-level definitions. Floating storage conversion and exact-width
scalar encoding are local primitive boundaries. `FloatCodec` relates one stored
scalar to its decoded comparison representation only on values reached from
FP32 casts or padding. FP16/BF16 rounding remains explicit. FP64 scratch is
FP32-origin, so its decoded comparison value has an exact FP32 representation;
its physical scratch width remains eight bytes.

The integer specialization interprets the source's complete safe conversion at
a named scalar macro port. `ConversionMacro.emitted_source` proves the original
source and emitted scalar arrangements agree from the two local zero-clamp
facts. `TypedIntegerMacro.source_native` executes the shared first-stage
quotation through both stores using that macro, and `Formats.decodeStored`
retains signed/unsigned width and boolean byte semantics. No record contains an
assumed final index, successful whole execution, whole-output equality, or frame.

Memory primitive rules, caller-provided entry descriptors, ordinary
`contiguous()` facts, and fresh allocations are reused from the historical
repair. This packet does not add a host implementation, allocator, device
resource, grid-product, or compiler implementation refinement project. Local
scalar primitive boundaries are declared assumptions, not deferred numerical
experiments. No finiteness restriction is added.

## Validation

The original candidate has saved scoped build receipts. The current IO
reorganization is checked separately with targeted builds of the new adapters,
both entries and `Misc/TensorBridge.lean`. Public endpoints pass statement-surface
and axiom audits; independent targets pass non-circularity checks. No proof
holes or new axioms are introduced.

The actual candidate JIT bodies passed 16 Triton CPU-interpreter cases: global
and dimension NaNs, ties, infinities, signed zero, multiple dimension tiles,
FP16 rounding, integer NaN conversion, and bool byte conversion. Input bytes and
output guards were preserved. The checker also confirms exact AST equality of
all non-JIT top-level code against the historical repair. Five offline CUDA
sm80 compilations passed and saved TTIR, TTGIR, LLVM IR, and PTX. No GPU execution
is claimed.

Evidence is in `.veritile-cache/kernel-evidence/triton_argmax_nan_first`.
The immutable delivery packet pins source files, current receipts, dependencies,
validation results, and compiler artifacts. Its initial status is frozen for
independent review; author validation is not reviewer acceptance.

Revision v2 addresses frozen-v1 review R1 by carrying the scalar bindings and
complete current-memory source execution into the public Tensor endpoints.
Frozen-v1 and its `CHANGES_REQUIRED` review remain immutable history. The Python
source, admitted domain, requested dtype semantics, Real proofs, and saved
numerical/compiler evidence are unchanged; this revision adds no experiment.


## Public IO organization

`Correct.lean` has the global and dimension source branches as shared
`ProgramIO.Implements` specifications with independent mathematical indices
and short proof calls. The optional ordinary theorems in
`Misc/NativeCorrect.lean` retain the complete two-stage native global source and
complete dimension grid, including every source loop, with independent native
indices. `FPEquiv.lean` records that no alternate complete implementation target
is connected.

`IO.lean` exposes the global input, scalar INT64 output, both scratch channels,
actual first-grid/second quotation and selected format. Native INT64 scratch
is an initialized logical input over the full `midBlock` gather range. Only
`blocks` first-grid entries are writable; padded tail cells retain the exact
frame. All thirteen original requested formats, every raw FP32 input word,
scalar codecs, positive/active launch conditions and the source's index-width
branch are preserved. The Real descriptor retains arbitrary original region
names and permitted aliases. `DimensionIO.lean` exposes the complete logical
three-dimensional input and all native output coordinates; empty outer/inner
grids remain admitted. The Real dimension program retains its original
arbitrary block sizes, base, PID and alias scope.

`Proofs/AccessSafety.lean`, `NativeSafety.lean`, `RealDimensionSafety.lean` and
`NativeDimensionSafety.lean` prove bounds for actual evaluated source accesses,
including selected-address rereads and every loop iteration. The public IO
adapters connect those safety facts to successful complete execution, output
values and the exact exterior typed-cell frame. They reuse existing source
execution/selection results. All operational composition stays in `Proofs/`.

The exact former Real and physical native statements remain ordinary theorems
in `Proofs/RealKernelResult.lean` and `Proofs/NativeKernelResult.lean`, including
integer precision erasure and the stronger default-FP32 readback. The original
`global_tensor`, `dimension_tensor`, `global_graph_tensor` and
`dimension_graph_tensor` statements remain in `Misc/TensorBridge.lean` unchanged.
`Proofs/PhysicalConnection.lean` explicitly proves that each complete new typed
schedule agrees at every output with the original physical source execution,
using the same byte state, scalar binding, requested format and source pair.
Original byte frames, codec scopes and caller observations remain intact.

The earlier format reorganization preserved Python, contracts and numerical
evidence. The separate exact-integer candidate below adds the current pair.

## Selected pair review (2026-10-21)

`FPEquiv.global_equiv` and `FPEquiv.dimension_equiv` now compare actual complete
original and shifted programs with `R=[]`. The global candidate specializes
`BLOCK_SIZE=1024` offset formation to a shift by 10, with the original formula
for all other block sizes. The dimension candidate specializes `BLOCK_M=128`
row-offset formation to a shift by 7, with a generic fallback. These are exact
integer address strength reductions; no measured performance gain is claimed.
The historical NaN-sensitive source remains separately defective and is not
used as either side of this pair.

`Kernels.lean` now contains complete ordinary readable source quotations for
all four floating specializations, the integer conversion-store projection,
the integer second stage and the dimension loop. Expanded sources preserve the
legacy macros definitionally; the campaign source-preservation check proves
all old/new complete program equalities by reflexivity. Integer conversion is
still the original named NaN0/clamp/truncate/width-store scalar projection,
with exactly the same binding and decoded scratch formats. There is no new
unchecked conversion claim. `CandidateKernels.lean` contains complete ordinary
first-stage and dimension quotations. The unchanged second stages are reused
transparently by the candidate schedule.

The global theorem retains all thirteen requested formats, complete first
grid/second stage, original source geometry and reduction-tree premises,
initialized INT64 gather window, arbitrary FP32 words, first-NaN/tie behavior,
all scalar conversions and scratch/output/exterior frames. The dimension
theorem retains the complete original 128-row/128-column grid and all N-loop
tiles, including empty outer/inner products. `Proofs/Pair.lean` additionally
proves exact source run and safety equality for arbitrary raw dimension block
sizes, bases and states; the fallback does not narrow the independent arbitrary
geometry `Correct.dimension` theorem. Both actual schedules are safe and
successful and agree on every active output with both complete frames.

`triton_argmax_shifted.py` is the complete executable candidate, preserving the
wrapper, requested dtype dispatch, launch segmentation, allocations and return
shape. `check_pair_source.py` restores only the two guarded integer shifts and
candidate names and checks exact whole-module AST equality. No finite-input
restriction, numerical rewrite assumption or numerical experiment is needed.
Existing native/software/Tensor results retain their names and prior scopes
in `Misc/NativeCorrect.lean` and `Misc/TensorBridge.lean`; primaries do not
import them.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
