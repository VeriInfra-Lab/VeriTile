# Direct FP32 chunked BMM forward repair

The kernel revision is described in [KERNEL_REVISION.md](KERNEL_REVISION.md).
Its public objective is one source/candidate grid invocation, the actual Tensor
payloads and output addresses, an independent numerical result, and memory
frames. The host and autotuning results below document the earlier contract
revision and remain optional support.

The returned tensor implements chunked `A @ B.T`, optionally filtered by literal
signed sequence-ID equality. Invalid rows/columns of the final partial chunk are
zero in the Real target. With `causal=True`, every strict upper-triangular cell
is zero. `Target.lean` states this complete target independently of the source,
tuning choice, finite floating encodings and initial output storage.

The original causal kernel does not implement that complete tensor: its guard
skips whole tiles, while surviving diagonal tiles still write upper elements.
With `A=B=ones([1,129,1])`, chunk 129 and `(M,N,K)=(64,32,32)`, cell `[0,0,0,1]`
becomes 1 and cell `[0,0,0,100]` retains the allocation's prior value (−777 in the
reproducer). Earlier autotuning trials can also populate skipped cells. The
historical conditional lower/active-region result remains valid, but is not the
complete causal mathematical target.

The accepted widened arithmetic kernel is retained verbatim, including all nine
autotune configurations, masked loads, causal tile skipping, sequence filtering
and dot/accumulator ordering. A separate final `_clear_causal_upper` pass writes
FP32 positive zero to every strict upper cell after trials and the chosen launch.
It uses the existing `(32,64)` configuration's grid, so it adds no launcher
product restriction. Original conditional `.contiguous()` expressions remain.
No input snapshot, clone, negative-view resolution or payload normalization is
performed.

## Admission and return contract

- A/B are matching FP32 tensors of shape `[B,S,K]` or `[B,S,G,K]`; output is FP32.
- Optional sequence IDs are INT64 `[B,S]`. Every signed ID is permitted,
  including −1 and −2; there is no ID-content guard.
- All used tensors are strided, contiguous, naturally aligned, on one CUDA
  device, have no unresolved negative flag, and have pairwise disjoint data
  spans. A/B last strides are 1. Nonempty A/B queried strides must be the exact
  canonical contiguous values, and nonempty sequence strides must be `(S,1)`.
  This explicit check matters because `is_contiguous()` permits other singleton
  strides. Actual queried strides are retained in the formal metadata/call.
- Chunk size is an actual positive Python integer (not Boolean); causal is an
  actual Boolean. `output_dtype` is `None` or FP32.
- Each used byte extent and the output byte extent is below `2^63`. For nonempty
  output, batch and chunk×group grid dimensions are below `2^16`, and each of the
  nine configuration grids has x below `2^31`. There is no total-grid-product
  condition or payload-value restriction.
- Empty batch, sequence or group returns the exact fresh empty allocation
  without launching. K=0 is admitted, including ordinary empty-input strides;
  the source reduction loop then executes zero iterations.
- The exact fresh output object, shape, device and FP32 dtype are returned.
  Original inputs and every byte outside the output data span are preserved.

`Contract.lean` contains the ordered error branches. `TensorBridge.python_checks_iff` proves
that acceptance is exactly the conjunction above. Ordinary allocation success,
representable backing spans and a fresh disjoint output are the allocator API
boundary; they are not numerical or whole-execution premises.

## Proof surface

`Correct.bmm_chunk_fwd_real` now states `Spec.Real ((io p c).Implements
(formula p))` through shared `AlgorithmProgramIO`. `IO.lean` exposes the actual
FP32 main source's mathematical projection, signed sequence channel, exact
active input windows, complete dense output and actual cleanup stages. The
main grid retains ordinary same-initial-memory semantics; cleanup receives its
actual predecessor state. Existing legal-configuration membership is unchanged.
Every source access is connected to the declared windows, alongside successful
execution, full independent contraction and the exact exterior memory frame.

`FPEquiv.bmm_chunk_fwd_execution` states the whole-schedule pair
`original rest p chosen ≡[[]] indexCommuted rest p chosen`. The shared
`ProgramIO.Implementation` view reuses the existing `ProgramIO.Equivalent`
proof, including both safe successful schedules, active typed output equality
and both exterior frames. `NativeCorrect.bmm_chunk_fwd_native` retains the
independent native `ReadProgramIO.Implements` result as an ordinary theorem;
`bmm_chunk_fwd_execution_and_native` also preserves the former conjunction.
`NativeIO.lean` exposes actual fresh-register launches in their original order,
the original source and integer-index candidate, typed read channels, all output
readback and the exact writable mask. All configurations and arbitrary original
region aliases remain admitted. The pure reference follows native K-block/MMA
order and the original cell-read convention if an earlier FP32 output write
aliases a later real/int input read. Unwritten outputs retain their original
tags/read values; observing them does not enlarge the writable frame.

The raw parameters retain the selected dense FP32 specialization and contain
no Tensor object, API dtype request or admission wrapper. Operational proofs
stay under `Proofs/`, including the unchanged full `KernelExecution.Raw.complete`
materialization invariant and complete typed-state equality. Source/candidate
trace safety is proved through the actual intermediate states. The candidate
changes only a pure integer multiply: its four-instruction prefix executes
identically and the remaining source is identical. There is no floating
transformation law or new numerical assumption. `Misc/Audit.lean` audits the public
entries and Tensor applications; `Proofs/KernelAudit.lean` audits the foundations
without importing the public readers.

`TensorBridge` retains the complete former Tensor and wrapper statements:
`bmm_chunk_fwd_real`, `bmm_chunk_fwd_tensor_real`, `bmm_chunk_fwd_kernel`,
`bmm_chunk_fwd_native`, `native_launches`, `bmm_chunk_fwd_if_local`,
`bmm_chunk_fwd_execution`, input bindings and physical word/frame helpers.
The old KernelExecution interface now directly instantiates the generalized
raw proof; the Real application shares the extracted raw main-grid proof and
the same complete cleanup theorem. Tensor binding and all original physical
output observations are preserved.

The optional `HostCorrect.bmm_chunk_fwd_real` in `Proofs/HostCorrect.lean`
constructs the earlier ordinary flat-memory Real
execution for every legal finite tuning trace and chosen configuration. It
proves every returned coordinate equals the independent contraction target,
with arbitrary Real inputs unrelated to finite byte decoding, and proves the
exterior memory frame. The final clearing pass has exact upper-coordinate
coverage, unique program/lane ownership and a full frame.

`TensorBridge.bmm_chunk_fwd_native` constructs the entire typed source graph execution
from original physical input bytes, including all trials and the chosen launch.
It proves materialized observations, exact little-endian FP32 output bytes,
return identity/metadata, exact zero upper bytes and the exterior byte frame.
`TensorBridge.native_launches` derives finite program/loop controls, masks, active pointers and
backed nonwrapping byte ranges from admission for every legal configuration.
The clearing program has a separate finite-to-logical event-list equality and
active byte-address proof. These structural results have no numerical premises.

`TensorBridge.bmm_chunk_fwd_if_local` transports output using only local literal, cast,
binary and fixed K8 instruction-coordinate laws at reached operands.
`K8Plan.lean` expands each consumed source dot into K/8 ordered steps, with
exact A[row,K]/B[K,column] operand selection, complete and unique K coverage,
instruction output coordinates, and the actual FP32 carry. The selected
families are MMA16×8×8 for the three two-warp configurations and WGMMA64×N×8
for the other six. `DirectFP.body_k8` connects the complete source body and its
actual old/new registers to this plan without numerical assumptions.

A numerical K8 atom takes only two `Fin 8` word vectors, one carry word, one
instruction output coordinate and its live bit. It takes no variable K, full
source tile or reduction result. Every emitted MMA carries its C operand;
all WGMMA scale-D predicates are explicitly true, including the first step
with the initialized zero carry. Induction proves transport through the complete
plan. Standalone pending-dot payloads are proved unobservable and raw storage
leaves are fixed identities. No split-dot/add versus fused-rounding identity
is assumed.

This is conditional numerical correspondence, not an unconditional GPU/compiler
proof or FP32 equality to exact Reals. NaNs, infinities, signed zeros, subnormals
and overflow are not hidden admission restrictions. A concrete machine's local
numerical primitives still need their stated correspondence evidence.

## Validation

`python bench/repairs/tritonbench_g/bmm_chunk_fwd/check_contract.py` passes:
**9 admissions and 51 rejections**, full causal outputs across tuning trials,
tails, signed IDs, K=0, empty dimensions, exact input/output identities and byte
frames, plus both original causal counterexamples. The arithmetic AST digest is
pinned to the accepted widened kernel. CPU arithmetic and CUDA launches are
mocked; this is source/control evidence, not GPU numerical conformance.

An offline Triton 3.7.1 SM90 compile of all nine FP32 configurations confirms
the listed TF32 instruction families and live carry selection; the receipt is
`compiler_k8_receipt.json`. This is compiler structure evidence for the recorded
specialization (causal/sequence flags false), not GPU numerical conformance.

All local Lean modules compile with clean axiom audits. Shared source/core,
original acceptance artifacts, historical checkout and progress ledgers were
left unchanged. See `SOURCE_BINDING.md` for the binding and trust boundaries.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/Audit.lean](Misc/Audit.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.


The source-only cleanup on 2026-10-22 replaces the selected `bmmChunkFwd32%`
whole-source macro with the complete ordinary `mainSource` and
`mainIndexCommuted` quotations in `Kernels.lean`. `bindMain`, `fp32Source`,
`indexCommuted`, the general typed source and the complete 32x64 cleanup retain
their full functions. The proof-only before/after replay under
`.veritile-cache/finish-pending-105-20261021/source-only/bmm_chunk_fwd/` checks
all five complete function equalities by `rfl`. This changes source readability
only: the accepted input domain, all precisions, nine configurations, sequence
filtering, causal control, full main/cleanup schedules, numerical boundaries
and existing Real/native/program-pair/Tensor declarations are preserved.
No proof-completion count or numerical evidence status changes.
