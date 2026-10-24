# dequantize_matmul

The canonical checked Python is `dequantize_matmul_repaired.py`, SHA256
`5e7805a331a03eb83a8dab77455a01d77958a79e182f6a1cb6f21468a1e79a9b`.
This kernel proof revision changes neither Python nor its guards. The original
benchmark's missing scale-stride defect remains distinct from the selected
checked revision.

The actual Triton kernel computes the complete FP32 scratch matrix
`fp_b[k,n] = FP32(signed8(B[k,n])) * scale[n]`. It has one FP32 multiplication
and no accumulation, FMA, saturation or floating-to-integer conversion. The
independent Real formula is `(B[k,n].toInt : Real) * scale[n]`. Real interprets
the source mathematically; no native-to-exact-Real equality is asserted.
The following `torch.mm(a, fp_b, out=c)` is an external library consumer.
The new public results prove the kernel only.

The existing checked contract requires contiguous strided resolved A/B/scales,
A/scales FP32 and B signed INT8, compatible dimensions, one CUDA device and
nonempty four-byte FP32 input alignment. It admits offsets, same-dtype read-only
aliases, empty dimensions and arbitrary inert singleton strides. Mixed INT8/FP32
raw-storage aliases remain rejected. All FP32 words remain admitted: there is
no finite/nonzero guard and zero multiplied by infinity is not simplified to
zero. Existing optional-output checks are unchanged.

The checked launch passes canonical strides `(N,1,1,N,1)`, so singleton Tensor
strides do not create new inactive-address constraints. K=0 or N=0 yields an
empty grid. For nonempty launches the existing K/N and extent guards imply
native signed64 coordinate and byte-offset bounds for all 17 autotune
configurations. The scale load keeps its wider N-only mask, including inactive
K tail lanes. Shared Triton runtime grid-product compatibility remains deferred
outside this per-kernel task.

`Kernels.lean` retains the complete readable typed quotation. Its candidate
only commutes the exact integer N-mask product; it makes no floating rewrite
or performance claim. `KernelExecution` and `KernelResult` execute every source
statement with an opaque floating carrier. `KernelGrid` proves complete output
coverage, unique active writers and the exterior typed-memory frame, including
empty and ragged grids. `KernelEquivalence`/`KernelWhole` connect both complete
successful launches to the same result. This generic chain also supplies the
mathematical interpretation without repeating a separate scalar proof.
The existing Real interpreter proof in `Proofs/Algorithm.lean` is preserved.

`Correct.kernel_correct` uses the shared `ProgramIO.ImplementsOn` judgment
for the independent signed-integer/column-scale formula. `FPEquiv.kernel_equiv`
now states `original multiply rest p ≡[[]] indexCommuted multiply rest p` for the two existing complete
typed sources. `Proofs/PairEquivalence.lean` reuses the existing exact integer
N-mask product execution proof. `NativeCorrect.kernel_equiv` retains the
original `ProgramIO.EquivalentOn` result for both actual full grids, the signed
INT8 input domain, safe traces and the exterior frame;
`FPEquiv.kernel_io_equiv` is its ordinary compatibility alias in `Misc/Compatibility.lean`.
Both reader entries have one public specification and a short composition proof.
`IO.lean` exposes the INT8 matrix, independent FP32 scale and FP32 output channels,
their full windows, and every actual launch. The logical signed-byte domain is
the image of `BitVec 8.toInt`; it introduces no numerical experiment assumption.

`Proofs/AccessSafety.lean` proves the actual source traces safe, including the
wider N-only scale load. `Proofs/IOGrid.lean` identifies the fixed shared
`runSchedule` with the existing full source grid, then reuses its execution,
independent outputs and exact frame. The public result covers all K-by-N outputs;
empty and ragged grids retain their original scope.

`Proofs/KernelRealResults.lean` and `Proofs/KernelNativeResults.lean` preserve the
earlier raw whole-grid statements, including the independent native signed
promotion/multiply result and output-ownership theorem. `KernelAddresses` reuses
the existing guard-derived `IndexABI` proof. Complete admitted Tensor/pointer
applications remain in `Misc/TensorBridge.lean`, deriving from the same source
execution foundation. `TensorBridge.python_checks_iff_preconditions` retains
exact Python admission.

`KernelTensor` uses shared `Torch.FP32.Tensor`, `TensorLaws` and `logicalValue`
for scales/output. The necessary INT8 input has shared Tensor metadata plus
ordinary byte-storage, physical-index and pointer observations. The initial
byte decoder derives the native input words, and contiguous API laws connect
source pointer arithmetic to actual storage offsets and physical indices.
Scratch freshness, physical byte separation, alignment and ordinary Tensor
metadata are explicit allocation/API boundaries, not extra caller assertions
or a proof of an allocator implementation. The source is the sole producer of
final typed cells. No new final-byteheap interpreter is introduced.

The offline probe compiles unchanged Python with Triton 3.7.1, CUDA sm80,
128×128, warps=4, stages=3 and an explicitly supplied runtime-i32 signature.
It is not a captured JIT binder trace. The observed profile has signed byte
loads, `cvt.s16.s8`, `cvt.rn.f32.s16`, one `mul.f32` per value, and
`st.global.b32`; 128 emitted store chains are retained in the PTX evidence.
`KernelModels` cites the official conversion, multiply and store rules. The
native formula preserves signed promotion and source operation order with an
explicit scalar FP32 multiply primitive. Primitive rounding, FTZ, conversion
and special-value semantics are model boundaries, not experiment TODOs.
Numerical rewrite assumptions and deferred numerical experiments: **none**.

The existing CPU AST check passes 45 complete output/metadata/frame cases over
all 17 configurations and 25 guard rejections, including offsets, ragged tails,
empty extents, unusual singleton strides and read-only input aliases. CUDA is
mocked and CPU torch.mm executes the external consumer. These are source and
contract checks, not GPU arithmetic measurements; no GPU run is claimed.

Historical whole-wrapper/consumer public sources remain in
`Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`. Their exact prior
contents and README are recorded by `legacy-manifest.json` in the evidence
directory. They are not new acceptance obligations. In particular, historical
TF32/FMA/library-lane interfaces do not become additional numerical experiments
for this kernel. New targeted receipts, current source/object hashes and reused
CPU/native evidence are recorded in the delivery packet. Whole-task acceptance
and independent replay remain the main agent/reviewer's decision.

`Correct.lean` and `FPEquiv.lean` are independent of `Misc/TensorBridge.lean`.
Supporting proofs import no reader entry. Python, the complete selected source,
input contract, native primitive profile and prior Tensor conclusions are unchanged.

## Selected pair review (2026-10-21)

The main pair uses the shared `ProgramIO.Implementation` view for both complete
actual launch schedules on the original signed-byte input domain.
`Proofs/PairEquivalence.lean` directly lifts the existing full-grid equivalence
certificate through that fixed view. It preserves signed INT8 promotion, FP32
multiplication order, all loads, casts, stores, source precision, positive block
and separation premises, safe traces, all output cells and both exterior frames.
The only transformation is the existing integer N-mask product commutation in
`Kernels.candidate`. `#print_fp_assumptions kernel_equiv` reports no experimental
rule. The previous ordinary IO relation remains in `Misc/NativeCorrect.lean` with
`FPEquiv.kernel_io_equiv` as a compatibility alias in `Misc/Compatibility.lean`; Real correctness is unchanged.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
Legacy native compatibility names are available from [Misc/Compatibility.lean](Misc/Compatibility.lean).


Function-body copy migration (2026-10-24): `TypedSource.kernel` contains the
original repaired Python body verbatim, including its docstring and both
`to(tl.int64)` program IDs. Pointer storage, seven runtime integer call values
and the two constexpr block sizes are declared outside the quotation.
`index_commuted.py` is explicitly newly authored Python for the existing Lean
N-mask candidate; it does not replace the historical original.
`Proofs/SourcePreservation.lean` checks the entire original function family
(all strides/counts/blocks) and both complete canonical programs against the
retained old programs. Full selected Real/FP, safety, output/frame, grid,
native and Tensor scopes are replayed from those actual bodies.
