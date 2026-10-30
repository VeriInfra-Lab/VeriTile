# Cache transform: checked kernel proofs

The selected production source is `cache_transform_triton_repaired.py`.
This revision preserves that Python file and every input check. The original
benchmark remains separate and is still diagnosed as defective.

## Independent target and source

Prompt mode concatenates cache rows `0 .. length-1` for each sequence. Decode
mode takes cache row `length-1` for each sequence. `mathematical` defines those
operations using finite row vectors; it contains no kernel execution, address,
row table or output state. Both outputs are fresh FP32 Tensors.

`Kernels.source` is the one readable typed DSL used by `Correct.lean` and
`Misc/NativeCorrect.lean`. It preserves the selected source: uint8 payload loads/stores,
int64 row-table loads, uint64 address arithmetic, 256 byte lanes and one
one-dimensional launch per nonempty output. There is no payload arithmetic or
new optimized variant. The native observation target is the independent
row/byte copy graph followed by raw FP32 word reconstruction.

`FPEquiv.lean` records status `missing_pair_target` and
declares no primary program-pair specification. No existing comparison program
implements this selected gather; the original task's different programs and
the preserved byte-to-formula result do not supply one. `Misc/NativeCorrect.lean`
retains the complete ordinary theorem `FPEquiv.kernel_native`, including its
original full name, statement and proof.

`KernelSource.launch` executes the complete canonical source with `START=0`
and unchanged program IDs. Its proof reuses the existing full byte-grid proof,
while proving that its general scheduling helper specializes to this exact
single launch. No grid segmentation is added or investigated.

## Public results

- `TensorBridge.cache_gather_kernel_real` proves each complete successful gather,
  the independent Real sequence formula at its actual output Tensor
  coordinates, the full tagged-cell frame, exterior physical frame and
  preservation of its live input-cache and row-table windows.
- `TensorBridge.cache_gather_kernel_fp32` proves the corresponding result from
  that call's actual input bytes and independently created int64 row table,
  with exact FP32 words. All bit patterns are included: signed zeros, subnormals, infinities
  and NaN payloads. There are zero numerical experiment deferrals.

The Real interpretation uses symbolic source-word **addresses** as payload
labels, repeated across the four byte lanes of each word. Those labels are
interpreted through an arbitrary Real memory. The unchanged source copies
labels without inspecting them, so its actual structural execution proves
arbitrary Real copy results without encoding Reals into finite uint8 values.
These labels are explicitly a mathematical interpretation, not native bytes.
The native theorem independently runs that same DSL on actual uint8 values.
The old `Ideal.call` theorem is retained as historical work and is not the
public source-execution argument.

## Actual Tensor and memory binding

`KernelBinding.Inputs` binds the admitted cache metadata to the shared
`VeriTile.Torch.FP32.Tensor`, its actual storage, physical indices and observed
`data_ptr`. Logical int32 lengths are read at their actual strided coordinates,
including delayed negation; their values determine the checked request.
`Table` binds each generated int64 row list to its actual pointer and bytes.
The cosine and sine launches have independent table observations; no shared
physical table is assumed.
All table observations and bounds are conditional on nonempty work.

`KernelBinding.Output` contains an actual output Tensor object, dtype,
shape, storage, contiguous coordinates, device and pointer laws.
`Single.Allocation` separates that call's output from its live cache and table.
Each invocation has its own heap, Cache, Table and Output; no relation between
the two invocations' temporary addresses is imposed. The public theorems
compose the proved source result with byte-window realization at the output
pointer and then read its logical coordinates.
This is a source-memory binding, not a proof of Torch's allocator or GPU code.

`TensorBridge.cache_transform_kernel_real` and
`TensorBridge.cache_transform_kernel_fp32` in `Misc/TensorBridge.lean`
are conditional paired corollaries under the stronger `KernelBinding.Allocation`
premise, which protects both tables throughout both calls. Python may release
the first `_gather`'s temporary table before allocating the second output,
and an allocator may reuse its bytes. Consequently these paired corollaries
are not the general acceptance entry. The independent per-gather theorems
cover that execution without modeling allocator lifetimes or host sequencing.
`TensorBridge.python_checks_iff_preconditions` exposes the unchanged input-check
correspondence.

Cache loads and output stores are uint8, so no FP32 alignment guard is added.
The created int64 row table has ordinary constructor alignment on its observed
pointer. Existing native safety lemmas cover active and masked-tail indices,
including the unmasked row-table load after `safe = 0`. Data-pointer/window
lemmas connect finite addresses and nonzero storage offsets. Memory-model
rationale and API references are in `documents/GpuMemoryModel.md` and
`VeriTile/Torch/FP32.lean`.

The preserved input contract is rank-one int32 lengths and two rank-two,
contiguous, strided FP32 caches on the same CUDA device, with clear negative
and conjugate cache flags. Widths agree; row capacities may differ. Read-only
cache aliases and noncontiguous lengths remain supported. Prompt lengths may
be zero; decode lengths must be positive, and each length fits both capacities.
The existing extent and grid bounds are unchanged. `checkInputs_iff` gives
exact equivalence for the typed metadata/value interface; dynamic Tensor and
Boolean argument classification occurs before that interface.

## Original defects and retained work

The original prompt kernel reads a power-of-two cumulative-length vector
without masking: three sequences request four entries. Its unpadded
`tl.arange(0, hidden_dim)` also rejects a width of three. Shared stride
parameters cannot describe independent cosine/sine/cache/output layouts.
The already-selected repair addresses those cases through the checked row
table and masked byte gather. This revision does not newly broaden that repair,
copy inputs or normalize unsupported cache layouts.

Historical whole-wrapper statements remain in
`Proofs/HistoricalCorrect.lean` and `Proofs/HistoricalFPEquiv.lean` with the
old helpers. Required public imports do not include host orchestration.
`AdmissionCore` contains the reusable guard, signed-length and single-launch
facts; their existing declaration names remain unchanged. Shared original
`Proofs/IndexRewrite` and other reused helper APIs are untouched.

## Validation boundary

`check_contract.py` executes the actual Triton source through its CPU
interpreter, with CUDA metadata/context explicitly mocked. It covers
prompt/decode, byte tails, storage offsets, input aliases, singleton strides,
empty dimensions, raw FP32 classes, noncontiguous and delayed-negative lengths,
fresh output windows, guard sentinels, complete input backing-storage frames
and rejection before launch. This is interpreter replay, not GPU execution.
Compiler/runtime realization remains an explicit boundary; no new native
GPU result or floating primitive conformance claim is made.

`Correct.kernel_real` and the auxiliary `FPEquiv.kernel_native` retained in
`Misc/NativeCorrect.lean` use the shared
`ProgramIO.ImplementsOnCells` judgment. `IO.lean` displays the exact typed source,
int64 row-table and uint8 byte channels, all input/output windows, complete grid
and independent logical row-gather formula. The source execution and exact
exterior frame follow from the retained whole-grid proof. The explicit logical
domain preserves its geometry, metadata-range and fresh-output conditions;
empty launches acquire no metadata requirement.

`Proofs/AccessSafety.lean` checks the actual source load/store trace, including
the unconditional metadata read in inactive lanes. `Proofs/IOSafety.lean` and
`Proofs/IO.lean` derive every intermediate grid trace from the original typed
input bindings and declared capacities. Exact-cell input mode preserves the
original signed metadata and byte-cell tags without adding Python guards.

The former raw mathematical/native source theorems remain in
`Proofs/RealEndpoints.lean` and `Proofs/NativeEndpoints.lean`. This includes the
arbitrary Real observation of copied bytes. `Misc/TensorBridge.lean` retains both
complete per-gather Tensor applications and optional paired applications,
including physical output and input-backing frames. No numerical experiment,
new source implementation or grid-overflow compatibility work was added.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.
