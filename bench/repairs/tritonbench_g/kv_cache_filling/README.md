# Paged KV cache filling

Optional proof entries are now collected under `Misc/`: [KernelOnly/Misc/TensorBridge.lean](KernelOnly/Misc/TensorBridge.lean), [KernelOnly/Misc/NativeCorrect.lean](KernelOnly/Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

<!-- current-isolated-kernel-entry -->
## Current kernel-only revision

Accepted after complete consumer compilation, independent review and root import checks. Policies 0/4/8 complete shared source and candidate executions, independent arbitrary-Real and native targets, output readback and frame. Native payloads, INT32 metadata and INT64 job arrays bind to one caller byte heap; both versions produce the same source-derived post-memory.

The [previous verified entry](verified-entry.json) records acceptance of the
prior isolated generation. The 2026-10-21 source-pair review and current hashes
are recorded in `.veritile-cache/fpequiv-pair-review-20261021/group07-result.json`.

The selected policy 0/4/8 source and supporting proofs remain frozen in `kv_cache_filling_shared_revision002`. The current review entries are [KernelOnly/Correct.lean](KernelOnly/Correct.lean) and [KernelOnly/FPEquiv.lean](KernelOnly/FPEquiv.lean). Their shared [KernelOnly/IO.lean](KernelOnly/IO.lean) gives finite typed channels for the two INT64 scheduling inputs, complete key/value input rows, complete cache output rows and both scale/zero lanes. Policy 0 has FP32 cache outputs and no active metadata stores; policies 4/8 preserve exact integer byte payloads. Each invocation executes the actual shared source and installs only program-ID axis 0.

`Correct.whole_shared_real` covers every `j ∈ jobs r`, both source versions and every program ID. It proves the independent full-head mathematical formula through the shared `ProgramIO.ImplementsOn` judgment. `FPEquiv.kernel_native` now states `original ≡[[]] indexCommuted` for every scheduled job and every program ID. The source-bearing `ProgramIO.Implementation` view reuses `ProgramIO.EquivalentOn`, including both safe and successful actual executions, every active typed output cell and both exact exterior frames. Only the existing integer address product is exchanged. The former IO statement remains the ordinary `FPEquiv.kernel_native_io` theorem in `KernelOnly/Misc/NativeCorrect.lean`. Both versions are separately proved to implement the same independent native row formula in `Proofs/IOCorrect.lean`; no successful execution or desired output is an input premise. Actual source `TraceSafe`, including scalar scheduling loads, padded quantizer reads and metadata stores, is proved in `Proofs/AccessSafety.lean`.

The input domain fixes only the two decoded scheduling integers. Logical payload arrays are finite; [Proofs/RowCongruence.lean](KernelOnly/Proofs/RowCongruence.lean) proves that the source formulas need only those active coordinates. Jobs retain independent entry states, matching the previous complete family theorem; this interface does not invent a sequential global-grid execution. The exact previous arbitrary-state family statements, including all native outputs and equal concatenated store lists, remain in [Proofs/RawCorrect.lean](KernelOnly/Proofs/RawCorrect.lean) and [Proofs/RawFPEquiv.lean](KernelOnly/Proofs/RawFPEquiv.lean), based on the same `SharedExecution.executed` and exact source observations.

[KernelOnly/Misc/TensorBridge.lean](KernelOnly/Misc/TensorBridge.lean) retains all previous Tensor results as ordinary theorems. `TensorBridge.Native.whole_shared_native` applies the retained raw family helper, preserving native readback, byte-frame and equal-post-memory conclusions. Metadata strides remain actual `Tensor.stride` query observations in this bridge. Build the selected live entries with the isolated [build driver](../../../../.veritile-cache/public-io-format-20261020/kv_cache_filling/build.py). [verified-entry.json](verified-entry.json) records current hashes and validation; frozen generations and historical evidence remain unchanged. The new shared IO modules are an additive isolated snapshot with explicit `minimumFloat`/`reduceMin` child-safety rules, without promoting minimum support into the default core.

[Kernels.lean](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/sources/bench/repairs/tritonbench_g/kv_cache_filling/Kernels.lean) · [Contract.lean](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/sources/bench/repairs/tritonbench_g/kv_cache_filling/Contract.lean) · [Correct.lean](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/sources/bench/repairs/tritonbench_g/kv_cache_filling/Correct.lean) · [FPEquiv.lean](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/sources/bench/repairs/tritonbench_g/kv_cache_filling/FPEquiv.lean). The [versioned entry and hashes](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/verified-entry.json) bind all four files. The [consumer PASS receipt](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/consumer-build-receipt.json) covers all 40 task targets and 83 repository modules; the [shared PASS receipt](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/shared-build-receipt.json) and [shared acceptance](../../../../.veritile-cache/minimum-staging/root-review.json) identify the reused generation. The [independent final review](../../../../.veritile-cache/independent-reviews/kv_cache_filling-minimum-resume_attention_score/final-review.json) accepted this generation with no open findings.

Run the [pinned replay command](../../../../.veritile-cache/task-deliveries/kv_cache_filling_shared_revision002/replay-command.txt) from the repository root. Replay pin/import preflight passed; an additional full replay was not run. Native scalar/reduction operations and byte codecs remain primitive boundaries. This revision adds no numerical experiment or numerical deferral; the live default Lake graph and historical proofs remain unchanged.

## Historical contract and host review

The retained text below describes the earlier review. Its public-entry paths, open items and numerical-deferral classification are historical; the current selection and exact proof/model boundaries are recorded above and in `verified-entry.json`.

The repaired function fills caller-owned paged caches and returns `None`.
For batch `b`, token `i < q[b]`, and head `h`, it reads state row
`(start[b] + i, h)`. With `position = total[b] - q[b] + i`, the destination is
`(table[b, position // block], position % block, h)`. Every other backing byte
is preserved. Duplicate destination tokens are rejected before any write.

The unchanged original can include undefined padded lanes in its min/max
reductions, divides by zero on constant quantized heads, clamps missing page
table entries to a different page, and does not establish unique destination
ownership. The separate repair validates the metadata, constructs exact int64
source/destination job tables, repeats a real lane in reduction padding, and
uses scale one for constant heads. All pointer products are widened; masked
pointer lanes are clamped to a valid lane. No state or cache payload is copied.

The explicit supported contract is:

- FP32 state tensors `[tokens, heads, dk/dv]`; cache tensors
  `[pages, block, heads, ck/cv]`. Heads, block, and head widths are positive.
- Policy 0 uses FP32 caches. Policy 8 uses uint8 caches of full head width.
  Policy 4 packs the first half into the low nibble and the second half into
  the high nibble, so `dk = 2*ck` and `dv = 2*cv`.
- Policies 4/8 require FP32 scale/zero tensors `[pages, block, heads, 2]`.
  Policy 0 ignores optional scale tensors.
- All active tensors have strided layout on one CUDA device. Payload roles
  are contiguous, have no unresolved negative/conjugate flag, and have natural
  pointer alignment when nonempty. Nonzero storage offsets are supported.
- Metadata vectors and the page table use int32. Their logical coordinates
  are read with the ordinary Torch CPU/list API; metadata need not be
  contiguous. Signed starts/counts/totals and active pages are validated before
  natural-number scheduling. Inactive page entries are ignored.
- Writable backing allocations are mutually disjoint and disjoint from all
  read roles. Read-only roles may alias. Full backing byte intervals are
  checked, including separately wrapped overlapping external storage.
- Packed widths are at most `2**20`; payload byte extents are below `2**63`;
  the one-dimensional job grid is below `2**31`. Empty work skips allocation
  and launch. These are direct source limits, without a compatibility wrapper.

`Contract.lean` states the guards. `checkInputs_iff` and
`checkAdmission_iff` prove exact Boolean admission equivalence;
`admitted_metadata`/`admitted_page` prove that signed values are not silently
totalized to zero. `Kernels.lean` contains the ordered scalar source graph,
masked loads, reduction wiring, row stores, and schedule. The `Proofs/`
closure proves complete evolving-memory execution, native bounds, Tensor
pointer binding, ownership, readback, and the full byte frame.

The primary mathematical result is `Correct.whole_mathematical_real`: it
executes the ideal source grid/host over arbitrary Real input payloads and
proves the logical cache/scale values, quantized bytes, complete logical
frame, unchanged inputs and return behavior. It has no finite byte codec or
representation premise. The separate encoded observation helper is
`Correct.whole_tensor_real`. The production precision result is
`FPEquiv.whole_tensor_fp32`, and the unconditional structural
`Tensor.whole_tensor_source`. They retain the caller Tensor coordinates,
`None`, object identities, and restored CUDA device context. The Real theorem
is the mathematical interpretation of the source, not a claim that FP32
arithmetic equals real arithmetic. The FP32 theorem uses concrete binary32
operations and defined float-to-uint8 conversion. Policy 0 has no numerical
obligations (`copy_has_no_numeric_obligations`); its byte codec preserves all
FP32 words, including signed zeros, infinities and NaN payloads.

The existing theorems above remain conditional results about the local
scalar/event model. They do not yet establish successful execution of the
actual shared `ComputeKernel`, and they are not a completed kernel-only
revision. The existing `ScalarNumerics` and per-node
`Tree.PrimitiveAgreement` premises describe primitive semantic agreement;
they must be resolved by the source/model connection, rather than postponed
as numerical rewrite experiments. `TaskCastDomain` is also an additional
defined-conversion premise, not a payload guard in the Python wrapper.

The kernel-only revision has three concrete open items:

- Shared DSL/core support: the source's `tl.min(x, 0)` reduction is not
  currently represented. The available floating `tl.minimum` expansion uses
  comparison/select and needs a primitive preserving the source's NaN and
  signed-zero behavior. This is a modeling gap, not an original-source defect.
- Primitive binding: offline compilation of the unchanged repair with
  Triton 3.7.1 for sm80 emits `llvm.minnum.f32`/`llvm.maxnum.f32` and
  `min.f32`/`max.f32`. The existing `softwareOps` instead propagates every
  NaN at min/max nodes. Its local agreement premise is not automatic over
  the admitted FP32 domain. The emitted conversion is LLVM `fptoui` to i8,
  lowered here to PTX `cvt.rzi.u16.f32` and byte stores. Defined-domain
  conversion, backend behavior, and compiler-level undefined cases must be
  distinguished before claiming coverage of all admitted words.
- Cast-domain coverage: `Proofs/Counterexamples.lean` proves with Lean's
  kernel that, in the existing binary32 model, finite input `[0, 2^-149]`
  underflows the scale to zero for both policies 4 and 8. The first active lane then reaches an undefined
  input for the existing partial uint8 converter. Changing the arbitrary
  `undefinedCast` extension changes the active output byte. A NaN row gives
  another counterexample. These facts refute a universal derivation of the
  old defined-cast premise from the input contract; they do not by themselves
  establish a native GPU output or a Real-versus-FP equality requirement.

The selected native cast now has a separate, total model in
`Proofs/PtxCast.lean`. For finite FP32 value `q`, let `k` be its integer rounding
toward zero. The emitted `cvt.rzi.u16.f32` clamps `k` to `[0, 65535]`, and the
subsequent `st.global.b8` observes that value modulo 256. NaNs map to zero;
positive infinity maps to byte 255 and negative infinity to byte zero.
For example, 256 maps to byte zero, while 65536 maps to byte 255: replacing
the actual u16 conversion by saturating directly to u8 would be incorrect.
The model agrees with the prior partial u8 conversion on its defined domain.
No such domain restriction is imposed on this native profile.

The source's implicit uint8 packing arithmetic will be written explicitly as
`(q1 + q2 * 16).to(tl.uint8)` in the shared DSL. This expands an existing dtype
operation; it is not a new Python repair. `PtxCast.pack8` and
`PtxCast.pack16_low8` prove that both source uint8 operations and the emitted
16-bit shift/add produce exactly the same final low byte for arbitrary
operands. The packed intermediate has no branch, address, or metadata
consumer: the source passes it only to the byte store.

`Proofs/NativeProfile.lean` keeps native add/sub, minnum/maxnum, equality and
`div.full.f32` as explicit FP32 callbacks. In particular the PTX specification
describes `div.full.f32` as a full-range approximation with maximum 2 ULP error;
it is not the existing exact IEEE-rounded software division. The candidate
changes only integer address multiplication and must preserve all floating
calls and operands. Its proof therefore needs no arithmetic rewrite law or
primitive experiment. The native-profile module also connects the existing
ordered scalar helper to an independent native quantization formula. This
does not yet discharge the complete shared-kernel execution connection.

The uncompiled policy-4/8 source draft is retained as
`.veritile-cache/kernel-evidence/kv_cache_filling/QuantSourceDraft.lean`.
It preserves the Python helper evaluation order and the selected FP32,
uint8 and INT64 types. It still requires shared minimum/reduction support,
native reduction callbacks with their actual order, successful whole-kernel
execution, and composition with the Tensor/schedule/output/frame results.
The generic mathematical interpreters currently reject floating-to-integer
conversion, so a mathematical interpretation of the same quantized
`ComputeKernel` is also an explicit remaining model connection. The existing
ideal scalar/event theorem does not discharge that obligation.

Primary modeling references, read and captured with the offline artifacts:

- [PTX `cvt`: rounding, default integer saturation, NaN, and subnormal rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt).
- [PTX `st`: a wider source register stores only the low instruction-width bits](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st).
- [PTX `min`: default minnum semantics, signed-zero order, and subnormal handling](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min).
- [PTX `div`: `div.full.f32` approximation and separate IEEE-rounded modes](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div).
- [LLVM `fptoui`: representable truncated values and poison outside that domain](https://llvm.org/docs/LangRef.html#fptoui-to-instruction).

The total PTX profile describes the emitted native instructions for the
selected compiler/target. It does not assign a unique portable meaning to
LLVM poison or assert identical lowering for every compiler version. Compiler
and GPU implementation proofs remain outside this task's kernel scope.

The input guards and Python source remain unchanged. No finite-only guard
or experimental primitive-conformance TODO has been added. The conditional
historical proofs are retained. `Proofs/CopySource.lean` provides the readable
policy-0 specialization for continued implementation. `Proofs/CopyExecution.lean`
proves both source/candidate executions succeed, the complete K/V row values,
and the logical memory frame without numerical assumptions.
`Proofs/CopyBinding.lean::tensor_copy` composes these results with the same
admitted Tensor metadata, physical payload heap, actual INT64 job-array
pointers, and an input-only logical read law. Public kernel endpoints,
three-policy integration, and the remaining native/physical connections are
still pending; this branch alone is not a completed three-policy task.

Reproducible diagnostics are under
`.veritile-cache/kernel-evidence/kv_cache_filling/`: `MinReductionProbe.lean`
is an expected frontend failure; `native_probe.py` saves offline lowering and
hashes for one configuration per policy; `admission_probe.py` runs the
existing source replay plus explicit underflow/NaN inputs. This CPU replay
admits those inputs, but its policy-4 elementwise NaN behavior differs from
the saved native lowering. It is therefore not a substitute for native
semantics or GPU validation. No GPU execution was performed for this revision.
Ordinary Torch storage observations and CUDA launch realization remain
runtime boundaries, outside the kernel-only proof task.

`check_contract.py` executes the actual Triton source with its CPU interpreter
and a clearly labeled CUDA-metadata/context facade. It checks policies 0/4/8,
non-power-of-two padding, constant rows, history crossing a page, multiple
heads, offset caches, untouched backing bytes, empty work, and invalid input
rejection before launch. This is source replay, not native GPU validation.
The original benchmark Python file is preserved.
