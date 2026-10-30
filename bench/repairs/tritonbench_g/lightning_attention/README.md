# lightning_attention: direct forward inputs and complete autograd results

Reference responsibilities (2026-10-23): `KernelOnly/IO.lean` owns the logical channel accessors. `KernelOnly/Reference.lean` retains the independent Real formulas and coordinate-derivative assertion; its derivative proof is in `KernelOnly/Proofs/Reference.lean`.

This revision uses an explicit contiguous FP32 forward-input contract and
retains the original three Triton kernel bodies unchanged. It proves the full
causal linear-attention output and all three backward gradients. The original
benchmark remains untouched (SHA256
`e53de8325844dd0e093ab119dd5a7a83406ff8f10a41f3eab731ecf6b1fae96a`).

## Supported API and formula

Q/K have shape `[B,H,N,D]`; V has shape `[B,H,N,E]`. Inputs are ordinary,
contiguous strided CUDA FP32 tensors on one device, without unresolved negative
or conjugate flags. B/H/N are positive; D/E are powers of two at least 16.
Inputs may share storage, have nonzero storage offsets, or retain unusual
strides on singleton axes. The wrapper passes and saves their exact original
Tensor objects. It performs no Q/K/V copy or normalization.

The independent mathematical forward target, per batch/head, is

```
score(t,u) = sum over a < D: Q[t,a] * K[u,a]
O[t,c] = sum over u <= t: score(t,u) * V[u,c]
loss(Q,K,V,G) = sum over t,c: G[t,c] * O[t,c]
```

There is no exponential, decay or normalization. Backward returns fresh FP32
`(dQ,dK,dV)` with the respective input shapes. The Real theorem proves the
coordinate derivatives of this independent scalar loss, not just equality to
three named gradient routines. Forward returns fresh FP32 O and saves Q/K/V;
backward uses the current saved-input storage under ordinary autograd
version/lifetime semantics.

The later incoming cotangent G must have V's shape, device and FP32 dtype,
strided storage and resolved logical flags. It may be noncontiguous or
broadcast. The original `G.contiguous()` remains because the benchmark's
`O.sum().backward()` supplies a zero-stride cotangent. This is the sole retained
input materialization. It returns the same object when G is already contiguous.
G may alias O or saved inputs; gradient buffers are fresh.

Checks require both time/channel grid y extents at most 65535 and
`(B*H*N+128)*max(D,E) < 2**31`. The formed-offset envelope includes tail lanes,
forward advances and initial reverse pointers. Every actual launched FP32
input/output pointer is divisible by 4 and has
`pointer + 4*formed_limit < 2**63`. The wrapper selects the input CUDA device
for each launch and explicitly enables fusion. Empty B/H/N are outside this
positive-sequence contract; no empty support is claimed.

## Source and precision

Forward processes blocks of 64 and value channels of 16 or 32. It combines
causal within-block products with the carried K-transpose/V product. Backward
runs the complete intra grid, then the inter kernel's forward query-gradient
scan and reverse key/value-gradient scan, with subblocks of 32. Inter reads
the gradients actually stored by intra and its own current updates.

All input loads, computation and stores use FP32, with default TF32 tensor-core
dots. The native graph retains source update order, immediate left-dot/right-dot
seed fusion, explicit scalar outer carry additions, consecutive fixed-K8
instructions and the WGMMA seed-enable flag. Static `Plan` chooses the backend
and literal-zero seed behavior from matrix shapes only; it cannot inspect input
values or results. SM80 uses warp MMA; SM90 may use WGMMA and warp MMA.

The original backward mask creates only FP32 lanes 0..63 before subtraction.
`NativeMask` checks all 64 conversions and all 4096 subtraction/comparison pairs
in the concrete software IEEE profile using Lean kernel decision. Source
normalization proves signed causal comparisons and exact descending-loop
reindexing/fold order. These are structural source facts, separate from hardware
numerical conformance.

## Retained wrapper theorem entries

The current selected kernel entries are `KernelOnly/Correct.lean` and
`KernelOnly/FPEquiv.lean`; see [their scope](KernelOnly/README.md). Correct proves
the independent forward/VJP specifications. FPEquiv records the missing
comparison target. The following earlier wrapper theorems retain their original
statements and validation scope.

- `Correct.lightning_forward_real`: complete independent output formula,
  returned Tensor and shape/device/FP32 metadata, saved original Q/K/V objects,
  protected inputs and physical exterior frame.
- `Correct.lightning_backward_real`: complete intra/inter execution, exact
  gradient return tuple, independent finite-sum gradient values, actual
  `HasDerivAt` loss derivatives, original-input preservation and exterior frame
  through the later cotangent materialization.
- `FPEquiv.lightning_forward_native` and `lightning_backward_native`: complete
  typed-source execution and current physical FP32 words, static instruction
  graph, exact returned objects/metadata, saved descriptors, protected inputs
  and full physical frames. The native backward graph begins at its own later
  memory, then hands the actual intra result to inter.
- `FPEquiv.lightning_forward_bytes` and `lightning_backward_bytes`: explicit
  little-endian backing-byte inputs and returned byte memory, every FP32 output
  byte, exact word/byte correspondence, returned descriptors and saved context,
  and the complete exterior byte frame through the cotangent materialization.
- `FPEquiv.lightning_native_if_local` and `lightning_bytes_if_local`: complete returned-record equality between
  two instruction interpretations. The only numerical premises are scalar FP32
  addition and one bounded output coordinate of fixed-K8 WGMMA or warp MMA.
  There is no whole-dot, scan, tensor, VJP or final-memory numerical premise.

`Proofs/DirectInputs.lean` derives packed address readback from contiguous
layout, including arbitrary singleton strides, and proves the old helper's
`ForwardCopies` relation reflexively on the same current memory. No copy or
snapshot premise remains in the public forward result. `Proofs/Contract.lean`
models classified checks and derives the dimension, pointer-room and alignment
premises. Ordinary Tensor/allocator metadata, source descriptor identity,
actual pointer binding and successful selected-device execution remain explicit.
The typed `check_iff` theorem relates the check predicate and named contract;
Python correspondence is covered by the source/host tests, not a verified
Python parser.

`Proofs/NativeBytes.lean` supplies the exact four-byte codec, proves aligned
load-after-store and exterior byte preservation, and proves buffer publication
is a finite sequence of those stores (`write_stores`). `load_publish` connects
this byte execution to the existing physical word execution. `input_word`
derives each source input word directly from its four current backing bytes;
`live_byte_bound` binds those addresses to the checked Python `data_ptr` and
signed64 room. `Proofs/ByteHost.lean` composes this bridge with the complete
native source execution. The backward byte-copy contract describes only the
original cotangent input coordinates, its protected inputs and copy footprint;
no kernel result is supplied as a premise. Original forward input words and
saved input words in the byte endpoints are decoded from supplied byte memory.

The public FP layer imports no task Real proof or mathematical endpoint. Its
`SourceHost` executes the typed source directly; it does not assume that an
abstract runtime launch already equals the complete reference output. Historical
copying/runtime-bound helper theorems remain under `Proofs/` for preservation.
Compiler realization of the source-derived static plan and local instruction
routing is an ordinary compiler boundary. The saved profiles are finite
evidence, not a universal compiler proof. GPU instruction conformance and
FP32-to-Real accuracy are not claimed. Shared large-grid launcher compatibility
remains separately deferred in `bench/TODO.md`.

## Validation

All 70 task Lean modules compile under Lean 4.29.0, including independent
forward/VJP targets, exact native graphs, source-order/mask proofs, current-cell
readback and complete public endpoints. Standard-base axiom and noncircular
checks pass; there is no new axiom or `sorry`.

- `check_contract.py`: exact AST identity for all three original kernels,
  31 host admissions, 20 rejections, and seven complete torch-autograd CPU
  interpreter calls. Dyadic values match independent FP64 forward formulas and
  all three autograd gradients exactly. Includes N1/63/64/65/129, multiple
  heads/batches, the original B2/H8/N128/D64/E128 fixture, actual
  `sum().backward()`, broadcast cotangents, input byte frames, alias admission,
  4-byte-only pointer alignment and lazy-negative rejection controls.
- `check_cuda_lowering.py`: 18 actual-JIT-binder SM80/SM90 profiles for forward,
  intra and inter, including tails, the original fixture and offset pointers.
  Artifacts default to `/tmp/veritile-lightning-lowering`.
- `check_precision.py`: unchanged source ASTs, retained required cotangent
  materialization, independent FP dependencies and hash-bound TF32/K8 IR/PTX.

CUDA execution was unavailable. CPU interpreter and offline compilation do not
validate native GPU arithmetic.

## Kernel IO and Tensor interfaces

`KernelOnly/Correct.lean` exposes `forward_correct` and `backward_correct` using
shared finite IO and independent causal/VJP formulas, including coordinate
derivatives. `SourceIO.lean` names the channels and complete source schedules;
the shared judgment includes actual source access safety, successful execution,
all outputs and the exterior memory frame. `KernelOnly/Misc/NativeCorrect.lean`
retains the ordinary `forward_fp_equiv` and `backward_fp_equiv` observations using
`MaterializedProgramIO`, with
the unchanged scalar/fixed-K8 native instruction boundaries. Each CTA's stored
values are materialized before the next CTA reads them, including the backward
intra-to-inter transition. Original raw statements remain in
`KernelOnly/Misc/RawCorrect.lean` and `KernelOnly/Misc/RawNative.lean`.

`KernelOnly/Misc/TensorBridge.lean` retains the original `correct` and `fp_equiv`
Tensor applications with all original logical reads, byte frames, preparation
and address-safety observations. Its Real and native IO bindings connect the
finite channels to the original Tensor/byte memory; both interfaces reuse the
same source execution and existing store/readback results.


## Optional proofs in Misc, 2026-10-21

Optional native observations, Tensor applications and reader/audit entries are
imported explicitly from `Misc/`. The primary `Contract.lean`, `Correct.lean` and
`FPEquiv.lean` keep the same mathematical and program-pair scope. Main proofs do
not re-export optional native conclusions. Declaration names, statements and
proof bodies are preserved; this file organization changes no input domain,
precision, numerical rule or task completion status.
