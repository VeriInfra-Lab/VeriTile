# decay_cumsum: complete kernel execution and native FP32 graphs

Preserved raw endpoint readers are also in Misc: `Misc/KernelRealResults.lean`, `Misc/KernelNativeResults.lean`. Their full declarations are unchanged.

Optional proofs (layout review 2026-10-21): Native correctness, Tensor applications, and optional readers/audits are now imported explicitly from Misc. Declaration namespaces, theorem types/proofs, kernel-stage boundaries, input domains and prior completion/blocker status are unchanged. Main `FPEquiv.lean` no longer re-exports optional native theorems. Moved entries: `Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/NativeFPEquiv.lean`, `Misc/HostCorrect.lean`, `Misc/HostFPEquiv.lean`.

Function-body migration: `Kernels.lean` contains all six complete original and
existing mask-index candidate bodies, with 8/11/15 Python statements per
forward/prepare/backward function. Program-ID and loop-bound int64 casts,
pointer dtype casts, masks, branches and source order remain verbatim. Dtypes,
call arguments and the immutable module literal `inv_ln2` are bound outside
the pasted body. `jit-correspondence.json` checks the original function bodies,
call metadata and exact module literal; the candidates commute only the
existing integer lane-mask product.

The raw source uses signed indices. The reverse range retains `(BT - 1)` before
its int64 cast, including the empty `BT=0` range. The public repaired wrapper's
existing positive tile restrictions remain unchanged. The backward dq/dk
stores have no extra cast in Python and no invented cast in Lean. The ordered
FP reference retains these direct stores; the native symbolic carrier retains
their arithmetic provenance while byte encoding observes their raw FP32 words.
The historical hand-written AST is archived as a proof artifact, with no claim
that it equals the pasted source. Complete execution, safety, independent Real
results and the source/candidate equivalence are checked against the actual
source. The source pair uses no floating numerical law.

The canonical candidate is unchanged: `decay_cumsum_repaired.py`, SHA256
`8280a38504ba5e0d5fb398180cc30cb90230739a6ca545e2c455e1d6cc309974`.
All three numerical source bodies, original FP32 casts, FMA contraction,
approximate exponential, loop order and runtime scale argument are retained.
The original benchmark Python is unchanged.

## Independent mathematical targets

For one feature lane in a time chunk of length L, with entry-state inputs:

```
go[t] = sum(g[i] * 1.44269504, i = 0 .. t)
qg[t] = q[t] * (2**g[t] * scale)
kg[t] = k[t] * 2**(g[L-1] - g[t])
dq[t] = dqi[t] + dqo[t] * 2**g[t]
dk[t] = dki[t] + dko[t] * 2**(g[L-1] - g[t])
dg[t] = sum(dq[i]*q[i] - dk[i]*k[i], i = L-1 .. t)
```

Forward retains the source decimal approximation, rather than replacing it
with exact 1/ln(2). Backward writes dq and dk as well as dg; its scale argument
is unused exactly as in source. Prefixes restart at each complete or short
chunk. The Real reference uses independent finite sums. No general derivative
theorem or FP-to-Real error bound is claimed.

The independent native references operate on all raw FP32 words. Forward uses
FMA with the literal `0x3fb8aa3b`, including its first zero-carry step. Prepare
retains scale/exponential and q/k operand order. Backward retains fused dq/dk
updates, separately rounded dk*k, fused dq*q minus that product, the first
reverse carry's `gradient + 0`, subsequent `carry + gradient`, and approximate
FTZ exp2. No split=fused, reassociation, finite-input or positive-denominator
assumption occurs. Individual instruction policies, including NaNs, signed
zero, subnormals and FTZ, are explicit primitive model data.

## Existing checked domain and repair geometry

Original caller tensors have exact shape `(B,H,T,DK)`, contiguous strided FP32
CUDA storage on one device, no unresolved negative bit, and aligned nonempty
data pointers. Offset views and arbitrary cross-role caller aliases remain
admitted. Dimensions and tile sizes are Python integers, including bool under
Python's `isinstance` classification. B,H,T,DK are nonnegative; BT,BK are
positive; BK is a power of two no larger than 1048576. Scale is a Python float,
without a finite-value restriction. On nonempty calls, ceil(DK/BK)<2**31,
T//BT<=65535, and B*H<=65535. Empty axes return before allocation and launch.
`KernelContract.checkInput_iff` proves the exact typed guard characterization.
No Python guard or selected input domain changes in this revision.

The existing repair snapshots each logical read role independently into
zero-padded feature width `ceil(DK/BK)*BK`. This is necessary because prepare
loads its last-decay row without a mask. Main and short-tail launches retain
head stride T*width. Source program IDs and loop bounds are widened to int64.
The source's original supplied output identities, None return and ordered
copyback behavior are unchanged.

## Primary proof boundary

`IO.lean` exposes shared `ProgramIO` with full `[span, BK]` typed channels,
original row/chunk/feature address windows, all inputs and outputs, and no
scratch. The backward `dqo`/`dko` channels retain their in-place input/output
identity. `Correct.decay_cumsum_kernel_real` is one `ProgramIO.Implements`
headline for all three kinds and every complete main/tail program. Its target
uses the independent formulas above. `FPEquiv.decay_cumsum_kernel_equiv` is
an ordinary native `ProgramIO.Observes` theorem in `Misc/NativeCorrect.lean`, retaining the selected object's
`Binding`, full arbitrary native input Nodes, host scalar and runtime FP32 ABI.
`Native.scaleSlot` is the existing syntax marker interpreted through that ABI;
it does not specialize the runtime scalar to a numerical constant.

`IOSafetyPrefix`, `IOSafetyBodies`, `IOSafetyBackward` and `IOSafetyLoops`
prove actual source access safety from the declared windows, including the
unmasked last-decay load, both forward loops, signed reverse loop and every
in-place store. They reuse the existing execution invariants. `IOResults`
reuses `FPTile.complete` for successful execution, every emitted value and the
raw-cell frame; `IOCorrect` composes those facts into the shared judgments.
No execution result or safety condition is an input assumption.

`Misc/KernelRealResults.lean` and `Misc/KernelNativeResults.lean` preserve
the exact prior raw kernel/source/compiler declarations. `Misc/TensorBridge.lean`
retains all five original Tensor/check/address/loop-bound statements and their
existing execution foundations. Existing whole-grid coverage, independence,
allocation and copyback helpers retain their prior scope unchanged. No source,
Python guard, numerical law, compiler implementation task or GPU experiment
is added.

`KernelReal` connects the actual source result to the independent finite-sum
formulas. `KernelNative` connects it to the independent raw-word recurrences.
`KernelCompiler` consumes the same selected object's byte-decoded source-site
graphs during execution. The binding retains the canonical source hash, entry,
source template, actual pointer roles, FP32 payloads, runtime integer ABI slots,
BT/BK/DK constants, compiler defaults and fusion. Pointer and integer divisibility
attributes are selected-object metadata checked against the actual arguments;
no new fixed caller alignment is imposed.
Decoder recipes contain fixed control choices and symbolic argument slots;
the actual runtime arguments instantiate those slots afterward. The theorem
proves that this instantiated decoded source is the original typed source.
The original Python binary64
scale is converted by the selected object's local FP32 ABI conversion and used
by source execution, the independent reference and output readback. The source
scale slots are forward 8, prepare 11 and backward 14. A reserved
source-template marker denotes that runtime argument; it is never a numerical
constant scale=2. Object decoding and local instruction/conversion semantics
are explicit model interfaces, with no whole-execution or desired-output law.

`KernelTensor` binds the actual packed kernel Tensor arguments through shared
FP32 Tensor/storage API laws. Main pointers and interior short-tail pointers
are related to actual data_ptr values. Every active source read is inside the
original packed Tensor. Backward dqo/dko retain the same input/output objects.
The output placement contains only finite address coverage and separation;
it contains no success or output-value premise. The same constructed source
state is materialized at those exact output Tensor addresses, with native
logical readback, preservation of read-only inputs, and every byte outside the
complete store footprint. Distinct scratch roles are separated where source
stores could otherwise overwrite their reads; this is the existing packed
argument layout. No allocator action or host final-byte-heap is a proof gate.

`KernelIndex` derives signed element offsets and 64-bit byte addresses from
actual FP32 Tensor storage capacity and the checked grid. It does not require
the old extra signed byte-allocation cap. The full packed Tensor supplies tail
bounds; no fictitious contiguous tail slice is assumed.

Historical whole-wrapper allocation/snapshot/copyback results are preserved in
`Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`; `Misc/NativeFPEquiv.lean`
remains their compatibility reader. They are separate from the primary kernel
entries. The primary statements use the actual packed operands at launch and
do not re-prove PyTorch allocation, copying or the driver implementation.

## Validation

The current source CPU replay checks exact numerical ASTs (erasing only the
widened integer casts), 14 rejected inputs, 26 phase/shape controls and 119
program executions. It includes empty axes, width/time tails, input/output
aliases, guarded caller storage, and the original head-stride/omitted-tail
counterexamples. CUDA allocation and launch are mocked, so this is not a GPU
numerical conformance test.

The offline reproducer in `.veritile-cache/decay-cumsum-kernel-profiles/`
compiles exact selected source bodies with Triton 3.7.1, CUDA SM90, four warps,
three stages, one CTA and fusion enabled. Its 21 main/tail profiles cover all three
kernels and four BT/BK/DK controls, preserving scalar-one specialization
(including stride_d=1 and tail T=1) and runtime FP32 scale. Pointer divisibility
specialization is not asserted. A separate set of 24 synthetic i32/i64 ABI
controls is retained as supplementary compiler evidence. TTIR, TTGIR, LLVM IR, PTX and cubin artifacts are hashed. Length-one
controls retain the first FMA, g-g, approximate exp2 and reverse add-zero.
These are bounded compiler evidence; no GPU execution or universal compiler
conformance is claimed. Public axiom and independent-reference audits use
Lean's standard base only.

Source readability cleanup (2026-10-21): `Kernels.lean` now owns six complete ordinary `triton` quotations for the selected FP32 forward, preparation and backward programs and their existing mask-index candidates. The `Original` compatibility aliases and generic optional storage sources remain available. `Proofs/LegacySurface.lean` records the previous selected bindings; `Proofs/SourcePreservation.lean` proves both complete source functions equal by kind cases and `rfl`, for every Call, Phase and scale, including all main/tail geometry. Numerical instructions, runtime scale binding, source APIs, public declarations and existing completion status are unchanged.
