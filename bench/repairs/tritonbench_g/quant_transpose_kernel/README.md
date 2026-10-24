# quant_transpose_kernel

The canonical checked Python is `quant_transpose_kernel_repaired.py`, SHA256
`46e8cc01024faa27ea3187bc2d1c70b412392cea65d0c4be00de276335820e7d`.
It preserves the original kernel, both duplicate autotune configurations and
wrapper. No Python or guard change is part of this kernel proof revision.

The existing contract admits positive rank-two strided FP32 tensors on the
current CUDA device, resolved negative views, nonnegative strides with one
stride equal to one, four-byte input-pointer alignment and the existing signed
int32 envelopes. Slices, transposes, broadcasts and storage offsets remain
supported. There are no finite/nonzero payload guards or implicit copies.
`Proofs/Contract.lean` connects Python admission and public premises;
`Proofs/Indices.lean` proves the checked int32 controls, masks, offsets and active
uint64 pointer arithmetic.

`Kernels.lean` contains the complete readable 128×128/G8 source. It retains the
one-dimensional grouped schedule, the rotated final partial group caused by
`pid % group_size`, masked loads, rematerialized coordinates, FP32 inverse,
two ordered FP32 multiplies, signed64 `llrint`, transposed INT8 store and masks.
The candidate only commutes the integer input-row address product. No floating
commutativity, reassociation, finite-value condition or performance claim is used.

The independent kernel target at output `[column,row]` is
`low8(llrint64(127 * (input[row,column] * inverse)))`. The inverse is an actual
kernel input. The wrapper's `abs().max()` and reciprocal are not new proof
obligations. `TensorBridge.kernel_correct` interprets multiplication in Real and
states nearest-even signed64 conversion with explicit range handling.
`Correct.normalized_nonzero` in `Misc/RealCorollaries.lean` records the supplied
nonzero reciprocal corollary; import that optional module explicitly.
`TensorBridge.kernel_equiv` retains the FP32 word carrier, primitive native multiply
and the pinned conversion profile below. Both endpoints cover complete actual
launches, all logical outputs, exterior typed-memory frame and the candidate.
`TensorBridge.kernel_tensor_equiv` additionally states those results using actual
shared `Torch.FP32.Tensor` logical reads for the input and prepared inverse,
and the INT8 output Tensor's physical indices. `Proofs/KernelTensor.lean`
derives the descriptor word array from the shared Tensor API. Its matrix-index
binding preserves strided, broadcast and storage-offset inputs; it adds no
contiguous-input condition or reciprocal assumption. Ordinary metadata,
storage, pointer and byte-decoder bindings remain explicit model boundaries.
The complete new proof chain and both public endpoints pass targeted Lean
builds. Embedded axiom audits use only the standard base, and the independent
formula/input/result/store definitions pass non-circularity checks. Successful
receipts and current source/object hashes are pinned by the delivery packet;
independent whole-task acceptance remains with the main agent/reviewer.

The existing llrint adapter executes actual source statements. At dynamic
pointer stores its logical integer cells retain the signed64 register. The
public byte observation takes the low eight bits for the INT8 Tensor; it does
not mislabel the output as a signed64 allocation. Initial typed reads connect
to the existing Tensor/storage model and physical input/inverse bytes. The
native-address theorem names the exact offsets used by the actual shared
source and reuses the proved checked integer arithmetic. Output observations
use the actual output Tensor's storage base and physical index through the
new adapter. There is no new final-byte-heap interpreter or
host/allocation implementation proof.

The offline source probe in
`.veritile-cache/kernel-evidence/quant_transpose_kernel/native-probe.json`
compiles the unchanged Python using Triton 3.7.1, CUDA sm80, warps=4, stages=3
and an explicitly supplied runtime-i32 signature. It is not an actual JIT
binder trace. The emitted sequence contains two ordered FP32 multiplications,
`cvt.rni.ftz.s64.f32` and `st.global.b8`. `KernelModels.nativeRound` records
nearest/ties-even, FTZ subnormal input, signed64 destination clamping, NaN to
signed64 MININT and signed infinities to the respective bounds. In particular,
NaN's low byte is zero. These rules apply to this emitted profile; no universal
libdevice invalid-domain claim is made. Official references are cited next to
the definitions:

- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
- https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st

Scalar primitive and conversion semantics are model/proof boundaries, not
numerical experiment TODOs. Numerical deferrals for this revision: **none**.
The exact `zero_infinite_inverse` calculation uses `FP.mul .ieeeValue` followed
by the pinned conversion; it does not identify arbitrary native multiply
callbacks with that software profile and is not a GPU observation.
Zero, NaN and infinity remain admitted. The all-zero host input has no ordinary
Real reciprocal; the preserved legacy outcome theorem records that boundary,
while the pinned native profile supplies its concrete conversion observation.
Older wrapper/outcome sources are preserved separately in
`Historical/LegacyCorrect.lean` and `Historical/LegacyFPEquiv.lean`; neither is in the new
public import closure or counted as a new kernel proof. The targeted
LegacyCorrect refresh passed. The optional LegacyFPEquiv refresh encountered a
stale import collision, recorded in `legacy-native-refresh.log` in the evidence
directory. Further historical refresh is not an acceptance gate for this kernel
target.

The current CPU contract check passes unchanged source/configuration/wrapper
AST checks, 11 source executions (including ragged/rotated groups and strided
views/broadcasts), 15 pre-host rejections, 3 propagated failures and explicit
forwarding of exceptional numerical inputs. These are not GPU arithmetic
experiments. GPU execution and independent replay are reported separately in
the eventual delivery packet.

The first packet, `delivery.json`, remains frozen. The Tensor adapter is
delivered in the separate `delivery-tensor.json` packet with fresh adapter
and public-endpoint receipts. The prior public source, documentation and
object family are preserved in `tensor-adapter-prior/` with a hash manifest;
source, CPU and native-profile evidence are reused unchanged.

`Misc/TensorBridge.lean` contains the ordinary actual-Tensor application theorem
and pointer/input binding helpers. The primary Real and native kernel
specifications remain in `Correct.lean` and `FPEquiv.lean`.

`Correct.kernel_real` states the independent mathematical byte formula through
shared `IntegerProgramIO.ObservesInteger`, uniformly for either retained index
version. `FPEquiv.kernel_native_equiv` compares `IO.originalProgram ≡[[]] IO.commutedProgram`
for the actual source pair. The shared program view retains complete successful
executions, access safety, fixed low8 output observation and exact raw exterior
frames. `Misc/NativeCorrect.lean` retains the independent native byte formula and
former combined observation as the ordinary `FPEquiv.kernel_native_observed_equiv` theorem. The headlines
contain source IO, geometry, region separation and primitive model parameters;
they contain no raw state, runner, final witness or memory-cell expression.

`IO.lean` declares the strided M×N input window, scalar inverse cell, dense
transposed output window, full grouped source schedule and fixed low8 view.
The public integer payload is the byte's unsigned 0..255 bit representation;
it does not change the INT8 output storage type. `Proofs/KernelModels.lean`
contains the independent mathematical/native byte functions; `IO.lean` holds
their Real channel adapter and `Misc/NativeReference.lean` the optional native
adapter. Input/input aliases, broadcast strides, empty geometries admitted by
the raw kernel theorem, all FP32 words
and both existing source versions retain their prior scope. The only region
separation remains input/output and inverse/output, exactly as before.

`Proofs/KernelSafety.lean` follows actual `QuantizeIntegerStage` statement
states and proves the masked strided input load, unmasked scalar load and
computed-pointer store obey the declared region bounds for both versions.
`Proofs/PublicIO.lean` reuses the existing whole-grid exact readback and frame;
`Proofs/ByteIO.lean` applies the shared fixed-width observation without changing
the source, raw integer interpreter cells or exterior frame. The actual INT8
pointer interpretation remains the documented low8 output boundary.
The optional complete native observation theorem is in `Misc/NativeByteIO.lean`.

The former raw headline propositions remain ordinary theorems in
`Misc/RawCorrect.lean` and `Misc/RawNative.lean`. `Misc/TensorBridge` retains its
complete original propositions and reuses the same `KernelWhole` foundation.
Historical reader-facing host wrappers were relocated unchanged to
`Historical/LegacyCorrect.lean` and `Historical/LegacyFPEquiv.lean`; they are
outside the selected public/supporting proof closure and were not expanded or
rebuilt for this formatting revision. Current `Proofs/` modules import no
reader entry, and neither primary entry imports `Misc/TensorBridge`.

Targeted format build receipts are in
`.veritile-cache/public-io-format-20261020/batch12-quanttranspose-final001.log`.
Python, source DSL and input contract are unchanged. No new numerical
assumption, GPU execution or independent native replay is claimed.

Source readability cleanup (2026-10-22): `KernelSource.quotedOriginal` and `quotedCommuted` in `Kernels.lean` display the two complete ordinary `triton` quotations for the existing selected 128×128/G8 pair. `Proofs/LegacySurface.lean` archives the previous macro; `Proofs/SourcePreservation.lean` proves equality of the full Bool-dispatched source function for every Parameters value by cases and `rfl`. Other configurations, APIs, grouped schedule, precision, quantization order, domains and existing completion status are unchanged.


Function-body copy migration (2026-10-24): `KernelSource.sourceKernel` and
`commutedSourceKernel` now accept the unmodified original/candidate Python
bodies, with original pointer/argument names and all rematerializations.
Call-side metadata supplies FP32 input/inverse, INT8 output, six runtime
int32 argument roles and three constexpr parameters. The public selected
128×128/G8 adapters retain every existing admitted stride and geometry.
`index_commuted.py` explicitly materializes the already existing Lean candidate;
it is newly authored Python, not a historical original reference.
The frontend now exposes the true implicit INT8 store conversion. Therefore
the old whole-AST identity is replaced with a complete execution connection
through the retained integer store interpreter in `Proofs/SourcePreservation`.
Both paths narrow the same signed64 result to the same INT8 cell; no floating
identity or whole-program assumption is added. Direct full Real/FP, byte,
complete safety/output/frame, native and Tensor proofs target the new body.
