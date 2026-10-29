# Scalar-base power on signed FP32 views

`Correct.lean` exposes the mathematical grid result, and `FPEquiv.lean` exposes
the source index-commutation equivalence. [Misc/NativeObservation.lean](Misc/NativeObservation.lean)
retains the auxiliary native `ProgramIO` result over logical arrays and signed
input/output views. `IO.lean` visibly
declares source identity, full signed offsets, typed FP32 channels, logical
masks, no scratch and the exact capped CTA schedule. Capacities, scalar ABI and
active footprint separation stay explicit; public statements take no raw state.
[Misc/TensorBridge.lean](Misc/TensorBridge.lean) retains the former checked applications
and `Checked.TensorBridge.checks_iff_domain`. The bridge uses the physical
access lemma in `Proofs/KernelAccess.lean`.

The independent mathematical formula is `out[i] = Real.rpow base (input[i])`.
The Real domain permits positive bases, zero with nonnegative exponents, and
negative bases with integral exponents. This is the mathematical interpretation
domain, not a Python value guard. The FP32 equivalence admits every scalar and
exponent word, including NaNs, infinities, subnormals and signed zeros.

The existing candidate changes only the integer tile-offset multiplication
`tile_id0 * tile_size0` to `tile_size0 * tile_id0`. FP32 loads, runtime scalar
conversion, the libdevice power primitive, output conversion and stores stay
the same. No new optimization or numerical power identity is introduced.

## Source and public proof scope

`Kernels.lean` owns complete ordinary readable Triton quotations for both original
branches and their index-commuted variants. The selected `DTYPE` is FP32.
Signed block pointers lower to `base + logical_index * stride` with the actual
logical boundary mask; the complete signed sum is formed before conversion to
a natural cell index. Arbitrary inactive load values remain admitted.
`Proofs/Source.lean` extracts the tile from this source and reuses the existing
signed tile proof. `Proofs/Kernel.lean::signed_source_lowering` proves successful
execution and complete memory/undef equality with the historical signed source.

- `Correct.lean::kernel_raw_correctness` proves full kernel-grid success, the
  independent rpow formula for every logical output, input preservation,
  the output-exterior memory frame and active cell bounds for both variants.
- `FPEquiv.lean::index_commuted_equiv` proves both actual sources succeed and
  agree on every typed memory cell, for arbitrary unchanged scalar primitives.
- `FPEquiv.raw_grid_fp32` in [Misc/NativeObservation.lean](Misc/NativeObservation.lean)
  specializes the complete grid to raw FP32
  words, with unchanged primitive results, input/exterior frames and active
  address bounds for either variant.
- `TensorBridge.grid_fp32_equiv` preserves the checked two-variant equality
  and physical-backing bounds; `TensorBridge.kernel_correctness` preserves
  the checked mathematical application.
- `TensorBridge.checks_iff_domain` connects the metadata checker to the exact
  input contract.
- `KernelAccess.kernel_native_access` exposes the reused signed32 index/mask
  and active four-byte pointer correspondence under that same contract.

Public entries concern kernels. Existing wrapper execution and returned-object
results remain optional in `Proofs/HostReal.lean` and `Proofs/HostFP32.lean`.
Those historical regional corollaries retain their stronger whole-backing
`StorageBindings` premise; it is absent from the public kernel contract.
No proof helper imports a public entry. Existing grid coverage, signed access
and memory-frame proofs are reused; there is no new compiler/runtime task.

## Checked input contract and memory

The checked Python source adds `_check_inputs` before the existing wrapper.
Its scalar helper, `StridedBuffer`, heuristics and both JIT branches are
unchanged. The original Python SHA-256 is
`142398199794dc7df724c341d3f86b1af60f20a9280b695efeac5e68d757471d`.

`pow_scalar_tensor_repaired.py` and `Contract.lean` require:

- A Python `float` scalar; matching rank-one Tensor/StridedBuffer views with
  integer `0 < n < 2**31` and signed-int64 integer strides.
- FP32 views and FP32 Tensor bases on the same CUDA device, with strided base
  storage and no unresolved negative view bit.
- Four-byte-aligned storage and view pointers, finite native allocation bounds,
  and both signed endpoint cells inside the backing allocation.
- Nonzero output stride and disjoint active input/output bounding byte spans.

The already-declared span guard is conservative: it rejects disjoint interleaved
footprints when their bounding spans overlap. Both stride signs, zero input
stride, arbitrary bounded gaps/offsets and separated views sharing storage
remain supported. No copy or materialization is added. Empty input retains
the original pre-launch `ZeroDivisionError`.

Metadata uses shared `VeriTile.Torch.TensorMetadata` rather than disconnected
dtype/layout/device/negative flags. The task-local signed view describes the
actual pointer's byte displacement from the backing storage.
`Proofs/Physical.lean` interprets both views in one physical memory indexed by
absolute FP32 cell addresses. Alignment and the complete signed pointer sum
are established before division and conversion to a natural cell index.
The existing endpoint/span checks derive cell validity, containment in each
actual backing span, and active input/output separation. `data_pointer` and
`cell_bytes` connect that interpretation to the queried pointer and signed
byte stride; the public frame preserves every inactive physical cell.

Backing storage objects may overlap with different base pointers. For example,
`x = a[2:7]` and `y = torch.from_dlpack(a[16:21])` have disjoint active spans
even though y's backing lies inside x's backing. This admitted case is covered
without a whole-storage separation premise or a new Python guard. The
zero-origin uint64 address range used by the reused native-index lemmas is
only a coordinate system; actual storage containment is proved separately.
The original finite signed-index and byte-address correspondence is retained
in `Proofs/WholeAccess.lean`.
Shared load/store and alignment rationale is documented in
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md).

## Numerical and validation status

There are **no numerical transformation assumptions or deferred experiments**.
The FP kernel theorem starts from the supplied FP32 scalar word. Python-float
argument conversion and unchanged libdevice power are ordinary model
boundaries, not new per-task numerical experiments. The optional historical
`ExactExtraction` and `ExactPower` interfaces remain available as model
refinements; neither is needed for index commutation or used as a hidden axiom.

The Python harness checks original JIT/class AST identity, six capped launch
partitions, four signed/broadcast/shared-storage layouts, a dynamic-loop
control, a DLPack overlapping-backing/disjoint-active regression with the
complete physical exterior frame, six scalar edge groups, two original failure
cases and 21 checked domain rejections. The adapters execute source control and memory behavior on
CPU; they do not validate native CUDA power numerics. GPU execution and
independent native replay have not been run.

Targeted Lean builds pass for the public entries, source/kernel/contract
helpers, preserved original entries and optional host results. The public
proofs and source lowering use only the standard Lean axiom base; the FP
assumption printer reports `none` for `index_commuted_equiv`.

The Real/native interfaces use shared `ProgramIO.ImplementsOn` / `Implements`.
The Real domain is a predicate on independent logical inputs. Actual source
trace safety is proved in `Proofs/TileSafety.lean` and `Proofs/KernelSafety.lean`,
covering both original/candidate versions and the full grid-stride loop. The
shared schedule is exactly identified with the existing `runGrid` execution;
typed outputs and the complete frame reuse the established `grid_result` proof.
The former raw statements remain ordinary helpers in `Proofs/RawCorrect.lean`
and `Proofs/RawNative.lean`. `Misc/TensorBridge.lean` retains both former checked
metadata applications and derives them through the same `grid_result` proof;
backing-storage aliases outside the active spans remain admitted. The ordinary
Real power domain and unchanged FP32 pow primitive boundary are preserved.

The public power interfaces take `SignedView.View` (region, signed base and
stride) directly. The internal launch adapter supplies no Tensor shape,
metadata or device argument; its view correspondence is definitional.

Optional proof entries are in [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeObservation.lean](Misc/NativeObservation.lean). Their declaration namespaces, full statements and proof bodies are retained; the primary entries do not import these optional readers.

Source readability maintenance (2026-10-22): the whole-source generator is
replaced by four complete ordinary Triton quotations: the one-tile and
grid-stride branches, each with its original/index-commuted tile offset.
Audit-only `rfl` certificates prove full function equality for all four branch
functions and both aggregate selectors, uniformly in signed views, scalar
argument, extent, tile and grid-stride count. Existing execution/lowering
proofs and fresh joint main/retained optional imports are checked in the
source-only cache. Precision, ABI, input scope and main pair are unchanged.
