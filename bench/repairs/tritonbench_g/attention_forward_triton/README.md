# attention_forward_triton separate repair

The selected source is `attention_forward_triton_repaired.py`, SHA256
`545872e280f84d19725d001872cee31bf0714f8aa520e1b5a17dbd09fec165b7`.
This kernel revision preserves that Python source and its admitted domain.

## Formula, domain and precision

The independent Real target is full, noncausal attention:
`sum_j 2^score(n,j) * V[j,d] / sum_j 2^score(n,j)`, where `score(n,j)` is the
96-term Q/K product times the query-block scale and key-block scale. The source
iterates over every key; the unused stage arguments do not impose causality.
For each active query there is at least one key, so the Real denominator is
positive. The source's initial `l_i=1` disappears because first-block alpha is
zero. Padded feature lanes contribute zero and masked key tails have zero mass.

Q/K/V have matching shape `[B,H,N,96]`, independent positive affine strides,
FP32 storage and distinct live allocations. Scales are FP32 and have canonical
nonempty shapes/strides `[B,H,ceil(N/128)]` and `[B,H,ceil(N/64)]`. Inputs are
ordinary strided CUDA tensors on one device with resolved negative/conjugate
view flags. The wrapper checks signed32 formed-index bounds, natural pointer
alignment, unsigned64 formed addresses, grid-y extent and the five saved JIT
specialization classes. Empty B/H/N has no launch and skips specialization.
The ordinary `empty_like` output is bound using its actual returned strides.

The compute tile is 128 by 64 with 128 padded feature lanes, of which 96 are
active. QK uses FP32 inputs/default TF32 instructions. Scores, online state,
reductions and final division are FP32; P and V are cast to FP16, the PV dot
returns FP16, and the final output is FP16. All FP32 input words, including
exceptional values, remain admitted. Real erases rounding and casts; no
native-to-Real accuracy bound is claimed.

## Source and proof organization

`Kernels.lean` contains the complete readable original and head-mask-hoist
quotations, with the Python helper inlined. `Source.source_memory` proves
execution of the original quotation,
including initialization, all key blocks, normalization, narrowing and masked
stores. `Whole.run_correct` proves current-memory execution over the complete
grid in any legal order. `Real.answer_real` proves the independent attention
formula. These existing results are reused directly.

`Proofs/Kernel.lean` connects those results at the pre-kernel memory boundary.
It does not introduce a wrapper invocation or allocator procedure.
`Kernel.InputContract` collects the existing Python checks;
`Kernel.OutputBinding` binds the actual output Tensor's shape, strides, region,
dtype and device without assuming an output value. `tensorOffset` uses that
descriptor's strides and is proved equal to the kernel's output address.

`IO.lean` exposes finite Q/K/V channels `[B,H,N,96]`, the two scale
channels `[B,H,ceil(N/128)]` and `[B,H,ceil(N/64)]`, and the complete FP16
output channel. Affine windows, exact source quotation, source-selected
precision, cleared-register launch ABI and arbitrary legal full-grid schedule
are explicit. There is no private scratch. Input windows may overlap: their
simultaneous bindings require the same physical read to have one value.
`logicalRead` only extends those finite arrays outside their windows so the
independent formulas can reuse ordinary total input readers; it cannot inspect
execution or final memory.

The main endpoints take geometry, input/output region separation and output
address injectivity. They expose no raw initial state or operational result:

- `Correct.attention_kernel_real` uses shared `ProgramIO.ImplementsOn` for the
  independent noncausal Real formula, including actual trace safety, successful
  full-grid execution, finite FP16-tagged results and the exact exterior frame.
  The `RealInputs` condition says the logical Real carrier contains finite
  reals; it preserves the old Real interpretation and adds no Python guard or
  numerical experiment requirement.
- `FPEquiv.attention_kernel_native_raw` is retained as an ordinary theorem in
  `Misc/NativeCorrect.lean`. Its shared `ProgramIO.Implements` statement observes
  the independent fixed-instruction native recurrence, with all FP32 words,
  actual safety, successful full-grid execution and exact frame.
- `FPEquiv.attention_head_mask_equiv` compares the two actual complete quoted
  programs, `original ≡[[]] candidate`, in every legal full-grid order. The
  candidate computes the active-head Boolean mask once before the key loop and
  reuses it in Q/K/V loads and the final store. It retains all FP32 input words,
  actual source safety, typed FP16 outputs and both exact exterior frames.
- `TensorBridge.attention_kernel_correct` and `TensorBridge.attention_kernel_native`
  retain their exact complete Tensor statements, including actual coordinates,
  FP16 output bytes, exterior byte frame and input byte-to-source-load readback.

`Proofs/SourceSafety.lean` proves the actual quotation's initialization,
unmasked scale reads, every 64-key loop predecessor and final masked stores.
`Proofs/PublicSafety.lean` connects those accesses to the finite IO windows and
the entire actual launch schedule. `Proofs/ReadCongruence.lean` and
`Proofs/LogicalInputs.lean` connect only the live typed input reads, leaving all
exterior memory unconstrained. `Proofs/PublicResult.lean` reuses the existing
`RawKernel.grid_result`, `Real.answer_real` and `NativeReference.target_binding`
results to assemble the public IO theorems. The exact previous raw statements
remain ordinary theorems in `Proofs/KernelRealResult.lean` and
`Proofs/KernelNativeResult.lean`; TensorBridge only redirects its imports and
one helper reference. No helper imports either public reader entry.

Historical host/checked theorems remain available under `Proofs/` and their
existing imports. They are reused where applicable, but host allocation,
return identity and the unused M allocation are not new proof obligations.

## Native semantics and assumptions

`Proofs/NativeReference.lean` states a separate recurrence over direct input-word
readers. It has no source state, registers, `Source.update`, `Source.accumulated`
or execution in its attention definition. `Proofs/NativeReferenceBinding.lean`
proves the complete score, reduction, update, loop and grid-coordinate connection
from the source-derived staged graph to that independent target.
QK expands to sixteen fixed K8 TF32 MMA components; PV expands to four
fixed K16 FP16 MMA components. Four physical row replicas retain the concrete
left-fold16/XOR2/XOR1 reduction trees. P uses each key's physical owner, and the
output selects its feature owner. Ordered score multiplies, standalone score
and alpha subtraction, FMA state updates, P/V narrowing, PV widening, division
and final narrowing remain explicit.

`Native.Instructions` supplies scalar and bounded-ISA semantics. The new native
endpoint does not assume `InstructionAgreement`, a complete dot, a whole result,
or successful execution. Historical agreement/transport theorems are optional
implementation-correspondence results; their old numerical-deferral wording
does not create an experiment TODO for this revision.

There are **no new numerical assumptions or deferred numerical experiments**.
Primitive arithmetic/ISA semantics, compiler lowering and Tensor APIs are model
boundaries. Any later quantitative approximation/error claim must state its
own target, precision and applicable data conditions. Missing program proofs
or source defects cannot be deferred as numerical assumptions.

`TensorStorage`, `OrdinaryPlacement` and `WordBinding` bind actual finite storage
and initial bytes. The native TensorBridge application also requires `RawInputs`: FP32 input
cells use the source model's `.real` load tag. This is a typed-memory binding,
not a finite-value condition. The proved `raw_read`, `source_load_word` and
`input_bytes` lemmas connect raw input bytes, typed `.real` reads and evaluated
FP32 source loads. The public `SourceReadback` conclusion uses the typed premise
through `source_readback`; mismatched cell tags cannot silently become zero.
Existing access/formed-address proofs cover masked lanes,
pointer advances, unmasked scale loads, all output coordinates and output holes.
The shared references are in
[GpuMemoryModel.md](../../../../documents/GpuMemoryModel.md#physical-byte-address-contracts).
[Tensor.data_ptr](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.data_ptr.html)
and [Tensor.stride](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.stride.html)
provide the corresponding ordinary API observations. Physical fragment/shared
memory routing, instruction realization, Torch allocation and GPU runtime
internals remain ordinary implementation boundaries. No universal compiler
correctness or device-resource suitability theorem is asserted. Grid-product
overflow and segmented launches remain deferred outside this task.

## Validation and retained history

The 2026-10-20 public-IO format pass runs only the targeted Lean builds,
public-surface and formula-independence audits, and source/object/dependency
receipt checks under `.veritile-cache/public-io-format-20261020/`. The selected
Python source, Kernels and Contract are unchanged. This pass does not repeat
CPU/compiler/GPU experiments or claim independent cold replay. The following
compiler/host evidence is retained from the earlier source revision.

The current kernel revision compiles the actual guarded source with Triton's
`create_function_from_signature` and `_pack_args`, preserving scalar-one
specialization and checking all scalar/constexpr values. The baseline and four
other saved pointer/integer classes use real CPU storage for metadata, with
CUDA allocation/context/launch mocked. The separate input strides, permuted
value view, naturally aligned but non-16-byte pointers, and aligned full tiles
pass the original wrapper before offline SM80 compilation. Artifacts are kept
under `.veritile-cache/kernel-evidence/attention_forward_triton/entry-jit/`
and `saved-classes/`. These are compiler checks, not GPU numerical experiments.
The unchanged `test_checked_contract.py` checks the actual wrapper, rejection
paths, empties and owning-device restoration. Targeted Lean receipts and the
final audit are recorded in the task delivery; independent acceptance is
reported separately. GPU execution and independent native numerical replay
have not been performed in this revision.

The original benchmark defect history remains separate. The repair uses
independent K/V/output bases and strides, correct K/V token advances, explicit
zero masked loads and a negative-infinity key-tail mask. It restricts the logical
head width to 96 instead of leaving the last 32 features of a 128-wide output
unwritten. The original FP32 caller's FP16-P/FP32-V dot mismatch is repaired by
the explicit V-to-FP16 cast. The original maximum seed was already negative
infinity; no finite-sentinel defect is claimed.

The selected 8-warps/2-stages source retains the existing resource configuration.
Historical 4-stage and 3-stage controls required 262156 and 196616 shared bytes;
the five current 2-stage compilations use 131076. These resource observations are
not universal device applicability or launch guarantees. Prior producer evidence
(`profiles004`, `native-graph-check001`, `Reader-build001`) remains historical;
this delivery does not claim to have independently replayed unavailable files.
The current Python hash also retains the owning-device context correction from
the prior `d38157443f0931243ca07de105681a46e8210d01ba16600adfc538b3437eefee`
revision. The source kernel graph and numerical domain were unchanged by that
host correction.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.


## Head-mask hoist pair (2026-10-21)

`TypedSource.hoistedHeadMask` is a related loop-invariant-code-motion candidate.
It adds `head_mask = offs_k < HEAD_ACTIVE` directly after the existing feature
index vector and reuses that mask at its four source sites. This removes the
repeated feature-index generation/comparison inside each 64-key iteration.
It does not alter floating operands, reduction order, TF32/FP16 MMA ports,
cast positions, storage precision, row/key-tail conditions, pointer advances,
output geometry or grid order. No performance improvement is claimed without
measurements.

The historical reciprocal candidate was considered first. It has FP16 Q/K/V
storage and predates repaired tail masking; its numerical rule additionally
requires finite actual accumulator/denominator operands and a nonzero
denominator. Those conditions do not follow for arbitrary FP32 input words.
That earlier conditional result remains preserved at its historical scope.
The current main pair uses the mask hoist to preserve the complete selected
all-word domain without a new numerical assumption or an input restriction.

`Proofs/HoistedSource.lean` adds the invariant Boolean mask to the original
loop context. `HoistedBody`, `HoistedLoop` and `HoistedOutput` prove actual
candidate execution against the same original independent recurrence.
`HoistedGrid` retains arbitrary full-grid permutations and complete source
frames. `HoistedSafety` proves each actual candidate access from the same
bounds, and `HoistedIO` assembles the shared typed IO judgment with the
unchanged independent native target. The main pair uses the shared
`ProgramIO.Implementation` interface and an empty experimental rule set.

Campaign evidence is under
`.veritile-cache/finish-pending-105-20261021/attention_forward_triton/`.
The focused source check verifies that the original complete declaration is
unchanged, both complete programs use the selected precision, the mask has one
definition and four uses, and inlining it recovers the exact original AST.
The original independent Real, native and optional Tensor statements retain
their existing scopes. This work adds no host, compiler or GPU validation
obligation and claims no native-to-Real equality.
