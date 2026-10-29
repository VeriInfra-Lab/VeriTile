# attn_fwd_causal: complete causal FP32-input attention

The selected canonical source is `attn_fwd_causal_repaired.py`, SHA256
`f63d95347c1900586aa74bee93d5b726962a4d9474737a64f3b3154bf52a8f67`.
This kernel revision preserves that Python source and its entire input domain.

The repaired source covers every one of the 128 output features, applies a true
causal mask, and runs the intended FP32 input ABI. It retains both helper calls,
the original block scales and the existing half-precision probability/value
product route. No input copies are introduced. The original benchmark is
untouched (SHA256
`c341c6af8eb93262c0c5aba48a61aac2b42a497e42f1d8068906d1d7ab189d48`).

## Supported call and independent target

`forward(q,k,v,q_scale,k_scale)` requires ordinary strided CUDA FP32 tensors on
one device, without unresolved negative/conjugate flags. Q/K/V share shape
`[B,H,N,128]` and canonical contiguous strides. Q scales have shape
`[B,H,ceil(N/128)]`; K scales have shape `[B,H,ceil(N/64)]`, also with canonical
strides. Nonempty inputs have disjoint full storage intervals. Canonical offset
views are supported, including pointers aligned to 4 but not 16 bytes.
The returned fresh contiguous FP16 tensor has shape `[B,H,N,128]`. The original
unused FP32 M allocation is retained.

The independent mathematical target, for an active row i and feature d, is

```
s(i,j) = (sum over e < 128: Q[i,e] * K[j,e])
         * Q_scale[i//128] * K_scale[j//64]
Y[i,d] = (sum over j < min(N,i+1): 2**s(i,j) * V[j,d])
         / (sum over j < min(N,i+1): 2**s(i,j))
```

Batch/head indices are shared throughout. There is no additional square-root
normalization. The Real projection erases rounding and source casts.

The checked integer envelope bounds every formed offset, including masked
lanes, pointer advances and scale accesses, below `2**31`; `B*H <= 65535` bounds
the second grid dimension. Nonempty calls check actual input pointer alignment
4 and output alignment 2, and `pointer + 4*envelope < 2**64` for every buffer.
Empty B/H/N calls retain allocation/envelope checks and return without querying
pointers or launching. Their input/output stride descriptors remain exact and
unconstrained. `Observations p` records the actual Python envelope; its equality
to the canonical model envelope is required only for a live launch. Thus an
empty-strided input is not silently subjected to a different canonical envelope.
`Proofs/EmptyReturn.lean` proves that the launch schedule is empty and that the
returned object is exactly the allocated descriptor, with no source execution. The fresh output allocator supplies its shape, strides,
separation and identity; it supplies no output formula.

## Repairs and source precision

The feature masks use the full head dimension 128 instead of 96, and masked
loads supply zero. The finite `-1e6` penalty is replaced by selection of `-inf`
for future keys and key-tail lanes. The diagonal helper ends at `min(end,N)`,
so every unmasked K-scale load addresses an existing key block. FP32 V is
explicitly narrowed to FP16 before the original FP16 probability/value dot.

QK retains default TF32 (`m16n8k8`) with FP32 result, followed by Q-scale and
K-scale multiplications. The online maximum, sums and state remain FP32;
exponentials use exp2. P and V are FP16, and the PV instruction result is also
FP16 (`m16n8k16.f16.f16.f16.f16`), before widening into the FP32 state.
Final output is FP16. Fusion is enabled explicitly. The launch uses
BM=128, BN=64, eight warps and two stages. The original four-stage configuration
exceeds SM80 shared memory; the retained accepted two-stage lowering fits.

## Kernel proof scope

`Kernels.lean` contains the shared readable FP32/FP16 quotation. Both helper
calls are inlined in order, restoring the caller's pointer values before the
diagonal call. Source parameters fix all Q/K/V/output strides to the same
canonical strides, so the separate base expressions in the quotation equal
the Python source's shared base. `Contract.lean` exposes the retained checked
input contract; empty tensors retain their actual unused stride observations.

The public kernel interfaces use raw geometry, output/input region separation
and a legal complete program order. `IO.lean` exposes five finite input channels,
their actual source address maps, the full `[B,H,N,128]` FP16 output channel and
the exact source launch schedule. There is no private memory or scratch output.
`RealIO.lean` and `NativeIO.lean` select the existing mathematical and native
primitive backends and pure output observations.

- `Correct.kernel_real` uses shared `ProgramIO.Implements` for the independent
  causal Real formula in `Reference.mathematical`.
- The optional `FPEquiv.kernel_native` theorem in `Misc/NativeCorrect.lean` uses the same shared judgment for the independent
  native word formula in `Reference.native`. This is source-to-native observation;
  it does not introduce an optimization or a native-to-Real equality.
- Optional `TensorBridge.causal_kernel_correct` and
  `TensorBridge.causal_kernel_native` retain the full checked Tensor applications.
  Their statements are unchanged. They use the exact prior raw results preserved
  as ordinary theorems in `Misc/RawCorrect.lean` and `Misc/RawNative.lean`,
  adding supplied-output Tensor coordinates, FP16 output bytes, the exterior byte
  frame and four-byte FP32 input readback.

The shared judgment proves actual source access safety, successful execution,
all typed outputs and the exterior raw-cell frame. `Proofs/AccessSafety.lean`
checks the evaluated prologue, both complete key ranges, reset and final stores.
It proves bounds for masked Q/K/V loads and output stores, and for the unmasked
Q/K scale loads. `Proofs/IOBinding.lean` connects these bounds to the finite
channels and proves that the shared launch schedule is exactly `runPrograms`.

The kernel domain retains arbitrary dimensions, empty complete grids, every
legal grid permutation and read-only input aliases. `Reference.inputAt` is a
pure finite-channel address lookup; simultaneous input bindings ensure that
overlapping ports agree. `Proofs/FiniteReadback.lean` proves that the complete
two-stage answer depends only on those finite input cells. Thus reuse of the
prior total-input mathematical/native lemmas adds no global memory assumption
to the public IO theorem.

`Proofs/KernelInterface.lean` derives finite regional capacity witnesses from
kernel geometry for the existing proof helpers. The main interfaces contain no
raw state, execution, per-address post-state, Torch metadata, checker admission,
Tensor bindings or allocator result.

`Proofs/Kernel.lean` reuses the existing `Source.source_memory`, full-grid,
Real-formula and finite-storage results directly at pre-kernel memory.
`Kernel.OutputBinding` supplies only ordinary output metadata; it does not
prescribe an output value. Output strides are constrained only for nonempty
launches, so empty strides remain admitted. No wrapper execution or allocation
procedure is part of the new endpoints. Prior `causal_real`, `causal_native`
and local-transport results remain available through `Misc/HistoricalCorrect.lean`
and `Misc/HistoricalFPEquiv.lean`; they are not new host proof obligations.

`Proofs/NativeReference.lean` states the causal native target using direct
FP32 word readers and two explicit folds: the prefix preceding the query tile,
then the diagonal tile through `min(tile_end,N)`. It keeps the causal selection,
fixed K8 TF32/K16 FP16 instructions, four physical row replicas, actual reduction
trees, cast positions and FMA carry order. Its attention definition has no
source state, `Whole.answer`, `Source.update`, `Source.accumulated` or execution.
`Proofs/NativeReferenceBinding.lean` proves the score, step, both folds and
coordinate connection to the source. It reuses `Native.input_from_current_bytes`
for typed input reconstruction. `RawInputs` requires the source memory's `.real`
FP32 load tag; all FP32 word values remain admitted, including NaNs/infinities.
This is a typed-memory binding, not a finite-value condition.

## Model boundaries and numerical classification

There are no numerical assumptions or deferred numerical experiments in this
kernel revision. No native-to-Real approximation or error bound is claimed.
Scalar FP32/FP16 operations, conversions and fixed-shape ISA instructions are
explicit semantic parameters, not primitive experiment TODOs. The optional
historical instruction-transport lemmas do not create numerical obligations.

Ordinary Tensor metadata, storage/pointer binding and successful fresh output
allocation are API boundaries. Physical register/shared-memory routing and
compiler realization are implementation boundaries; the source-to-independent
recurrence connection is proved. The saved SM80 profiles provide bounded
compiler evidence and do not restrict the public input domain. The historical
`CompilerBinding`/`SavedCompilerBinding` definitions remain available but are
not whole-result premises of the new kernel theorem.

Natural scalar alignment, unsigned address bounds and finite-storage frames
retain the shared [GpuMemoryModel.md](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts)
references. No universal compiler correctness, GPU resource suitability or
native NaN-payload guarantee is asserted. Triton runtime grid-product overflow
and segmented-launch compatibility remain deferred outside this kernel task.

## Validation

Targeted Lean builds and the final combined axiom/noncircular audit are recorded
in the task delivery. Existing Real, source, physical and host results are
retained; task-local cached objects were refreshed after an actual shared
`Structural.Algebra` layout mismatch. Independent acceptance is reported separately.

- `check_contract.py`: 32 admissions, 18 rejections, nine complete FP32 CPU
  interpreter grids with input-byte and output-hole frames. N includes
  1/63/64/65/127/128/129/257 and multiple batches/heads. Four additional empty-stride
  controls include `[0,1,1,128]` with strides `[1,1,1,1]` and a large empty view
  whose actual envelope is admitted while a fabricated canonical envelope would fail.
- The interpreter reproduces 4096 unwritten original feature cells and the
  original finite-penalty future-token leak using its working FP16 sample ABI.
  A separate tail control shows the original N=1 diagonal reaches nonexistent
  K-scale block 1. The repaired intended FP32 ABI passes these controls.
- Retained `check_cuda_lowering.py` evidence: eight current-source actual-JIT-binder SM80 profiles, including
  singleton/tail/prefix and 4-byte-only pointer alignment. It also reproduces
  the original FP32 mixed-P/V dtype compiler rejection. Artifacts default to
  `/tmp/veritile-attn-causal-lowering`.
- `check_precision.py` checks both helper phases, masks and source casts, then
  the hash-bound TF32/FP16 MMA, exp2, fused arithmetic and narrowing artifacts.

The current revision rechecks all eight canonical wrapper-to-JIT bindings
against those retained signatures, constexpr values and options, and verifies
the current-source artifact hashes. No compiler run is repeated merely to copy
history. `check_contract.py` and `check_precision.py` were run for this revision.
GPU execution and independent native numerical replay have not been performed;
CPU interpreter and compiler evidence do not establish either.

## Selected pair review (2026-10-21)

The primary `Equivalence.quotient_remainder_equiv` compares the complete
repaired source with `TypedSource.quotientRemainder`. The candidate replaces
`off_hz % H` with `off_hz - (off_hz // H) * H`, exposing the quotient already
computed for `off_z`. This is an integer CSE opportunity; no performance
measurement is claimed. The historical reciprocal candidate remains available,
but its separate floating-point connection is not needed by this pair.

`Proofs/PairConnection.lean` proves whole-state execution equality and actual
access-safety equivalence for arbitrary primitive backends using the exact Nat
remainder identity. `Proofs/PairIO.lean` connects both actual programs over every
complete legal grid permutation, all native input words, both attention stages,
tagged outputs and exterior frames. TF32 QK, FP16 P/V/PV and output, FP32 state,
masks, casts and instruction interpretation are unchanged. The primary pair uses
`≡[[]]`: it introduces no floating-point numerical rewrite or experiment.

The independent mathematical `Correct` target is unchanged. Retained
single-source native/software results live in `Misc/NativeCorrect.lean` as
ordinary auxiliary theorems, with their previous statements and domains.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
