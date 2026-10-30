Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

Current FPEquiv: `FPEquiv.affine_address_equiv` compares the complete original
source with the actual `token_attn_reduceV_affine.py` kernel/callable wrapper.
The candidate computes `(head*headStride + start*tokenStride) + offsets*tokenStride`
instead of `head*headStride + (start+offsets)*tokenStride`. This moves the uniform
origin to scalar arithmetic and removes one vector addition, while retaining
all FP32 arithmetic, reductions, masks, casts, grid order and stores. No measured
speedup is claimed. `Kernels.affineSource` contains the complete standard Triton
quotation and `Interface.affine` uses its actual invocations on the original IO.

The strict integer proof in `Proofs/AffineEquivalence.lean` proves assignment and
whole-state execution equality and equivalent source trace safety. It lifts
through every original grid invocation. The `≡[[]]` result includes both safe,
successful executions, active outputs and complete exterior frames on the full
original lengths/stride/alias domain, including empty grids and source block-zero
behavior. It introduces no finite-value or numerical assumption. The independent
Real target and optional native/Tensor results retain their prior scopes.

# token_attn_reduceV: explicit CUDA grid-y contract candidate

This separate candidate adds exactly two Python lines after the existing empty-batch return:

```python
if head > 65535:
    raise ValueError("query heads exceed the CUDA grid y limit 65535")
```

The accepted domain is exactly the preceding checked source's domain intersected with `B=0 or Hq<=65535`. Every previously admitted empty-batch input is retained, including `Hq>65535`. For nonempty batches, `Hq>65535` now raises before launch and before output mutation. All other guards, dtypes, shapes, content checks, stride choices, device context, operation order and wrapper return behavior are unchanged. The JIT kernel AST is identical. There is no copy, conversion, additional finite-value condition, positive-sequence condition, allocation-separation condition or segmented launch.

The predecessor selected source is `../token_attn_reduceV/token_attn_reduceV_repaired.py`, SHA256 `1182d7c84e373ae328250b0d8c63886d12f8bde5c36279bbcdc131db1de0e020`. Its frozen defect packet is `.veritile-cache/token-attn-reducev-kernel-evidence/delivery.json`, SHA256 `3448ed3f47f38734e163d1185a57d617b10f62359dd3936df5737481a04d645d`. NVIDIA CUDA Programming Guide 12.6.3 Table 21 gives maximum grid y of 65535. That packet preserves the actual admitted `(1,65536)` launch capture and official source excerpt. The predecessor files and packet are unchanged. This directory is a sibling because its original hash verifier fixes the original directory's recursive source-file set. Root owns review, canonical selection and ledger updates.

`Contract.lean` mirrors Python's checks on shared Torch TensorMetadata and the caller's signed-int32 words. `python_checks_iff` establishes success exactly for InputContract; `check_launch_iff` and `check_empty_iff` distinguish launch and empty paths. `exact_domain` and `empty_domain_unchanged` expose the precise domain change. Tensor type checks are represented by `checkArguments`. Rank, FP32/int32, device, no-negative/contiguous, extent, content and live pointer checks retain their original scope. No new layout tag check is imposed. Same-dtype input views may alias, including views in shared storage; mixed dtype overlap and output/read overlap use live view byte spans, not whole-allocation identities.

`Kernels.lean` reuses the unchanged single quotation in the predecessor `Source.lean`. `Proofs/Model.lean` instantiates every actual pointer region, stride, head group and launch parameter. `complete` proves successful execution of the quoted kernel over every permutation of the complete `(B,Hq)` grid, every output store and the complete source-memory frame. It requires no execution-success or final-result premise. The original dynamic 128-lane loop and final stores are executed even for rows with zero sequence length. `empty_call` gives the empty grid and preserves every caller word.

`Proofs/Tensor.lean` binds the shared caller Tensor object/storage identities, storage offset, physicalIndex and observed pointer through ordinary four-byte Tensor API laws. These laws cover FP32 and int32 storage geometry and assume neither a kernel result nor a layout conversion. Header supplies the checked dtypes and ordinary view flags. The same caller reader at a given physical address represents legal aliases. Active bounds and direct-address identities are derived; `active_logical` connects source numerical operands to the original logical Tensor cells. The public endpoints read back the caller's logical Out indices. `RawFrame` covers every region/index outside source stores; `observe` is the physical interpretation of exactly those stores in the caller output view, preserving its full exterior and every admitted input view.

The independent Real target is

`sum(n < length[b], P[h,start[b]+n] * V[pages[request[b],n],h/(Hq/Hkv),d])`.

`TensorBridge.kernel_correct` proves this target for every output together with actual quoted-grid success and both frames. `Proofs/Indices.lean` proves legal submitted x/y axes, signed32 grid product, padded P/page/loop arithmetic, active value/output indices, sign interpretation and aligned active 64-bit addresses. Masks may form unused addresses without dereferencing them. Tensor storage validity and absence of concurrent mutation are ordinary caller/API execution conditions; the Lean result does not certify a particular CUDA driver, compiler binary, resource availability or hardware implementation.

`TensorBridge.kernel_native` proves all output words against the independent `NativeTarget.reference`. A single caller word map supplies both numerical words and signed metadata. The target explicitly retains every masked operand, padded multiply, complete 128-lane reduction, FP32 accumulation step and final cast. Compiler choices are structurally restricted complete ordered trees and legitimate adjacent FP32 multiply/add contractions; they cannot inspect numerical payloads or supply a row result. Only scalar `addRN`, `mulRN` and `fmaRN` instructions are parameters. Their matching native ISA behavior, selected reduction/contraction profile and compiler/source correspondence remain the numerical/model boundary. No exact native-to-Real identity or universal GPU conformance is asserted. The candidate changes no numerical transformation and adds no scalar numerical experiment obligation. The predecessor's historical native instruction validation status remains separate from current transformation-rule admission.

`check_candidate.py` verifies sole-guard AST equality, executes the original 48-case checker against the actual candidate JIT in Triton's CPU interpreter, and checks live 65535/65536/65537 plus empty 65536/100000 boundaries by launch capture. The empty and rejected paths preserve the output. CPU CUDA metadata facades and capture do not execute a GPU. Historical eight SM80 lowering specializations are reusable evidence for the unchanged kernel only, not a universal lowering or numerical guarantee.

## Public IO interfaces

`Correct.lean` now states shared `ProgramIO.ImplementsOn` for the independent
weighted sum. `Misc/NativeCorrect.lean` states shared `ProgramIO.ObservesOn` for the original
source-ordered native recurrence. The Real main statement and optional native theorem retain every permutation
of the complete grid, existing output uniqueness and read/output separation,
source precision and full exterior frame. They contain no raw state, runner,
cell read or existential post-state.

`IO.lean` displays the six channels: runtime length/start/request metadata,
gathered token indices, probabilities and value vectors. Read addresses depend
on the actual metadata channels; all declared input aliases bind simultaneously.
The arbitrary logical `counts` match the runtime length channel. Their finite
supremum supplies padded logical shapes without adding a fixed maximum-length
input restriction. Empty grids mask unused metadata reads. Native `block=0`
keeps metadata loads and output stores but masks all loop-only input windows.

Native logical inputs retain arbitrary `Native.Term` payloads; a separate pure
observer produces FP32 words. This preserves every original typed word heap,
including a non-FP32 source cell whose typed read supplies a default term.
Restricting the inputs to addressed word leaves would change that scope because
compiler reduction/fusion choices can inspect expression shape. The actual
old heap injection, logical input binding and exact old reference specialization
are proved in `Proofs/Compatibility.lean` and `Proofs/NativeIO.lean`.

`Proofs/AccessSafety.lean` checks the actual metadata loads, every dynamic gather
iteration and the unmasked output store. `Proofs/Bindings.lean` derives those
bounds from public IO; `Proofs/Schedule.lean` follows actual predecessor states
and reuses existing source execution. `Proofs/Interface.lean` reuses `Grid.whole`
for complete outputs/frame, while `Proofs/RealIO.lean` reuses the established
weighted-sum identity. Former raw statements remain in `Proofs/RawReal.lean`
and `Proofs/RawNative.lean`. The reader audit is in the optional
`Misc/InterfaceChecks.lean`, outside supporting `Proofs/` modules.

`Misc/TensorBridge.lean` is unchanged, retaining all original Tensor/metadata inputs,
logical and native outputs, physical/input frames and empty observations.
Python, Kernels and Contract pins are unchanged. Current format verification is
limited to exact changed/new Lean targets, the unchanged Tensor bridge and the
task-root audit. Historical evidence is retained; no GPU, independent numerical
replay or new experiment is performed in this pass.


The complete original and existing affine candidate bodies now share the raw
source entry in the underlying task's `Source.lean`. `Kernels.lean` explicitly
forwards to that shared entry; it emits no replacement program or generated
body. The selected checked Python and existing affine Python remain unchanged,
and this directory's `jit-correspondence.json` checks both complete bodies.
Types, runtime call bindings and constexpr metadata stay outside the bodies.
The terminal `return`, all strides, gathers, masks, padding, reduction order,
FP32 arithmetic and output conversion remain. The existing FP32 input profile
and signed-int32 native metadata bounds are retained; unused generic dtype
helper parameters are removed without restricting any public theorem.

Both full executable statement lists are proved identical to their preceding
selected FP32 programs in `Proofs/SourceIdentity.lean` of the shared task.
Revalidation includes this directory's complete Real and FP specifications,
whole-state/failure and trace-safety affine equivalence, full-grid execution,
all physical frames and retained Tensor/native applications, together with the
underlying execution/host/native proof layers. No numerical assumption, input
restriction, new candidate or GPU experiment is introduced. Earlier checks
above remain historical; the current frozen-source receipts and unavailable
tool attempts are recorded separately in the task ledger.
