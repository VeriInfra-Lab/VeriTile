# context_attn_llama: kernel-only revision

This package proves the unchanged checked FP32 page-gather kernel in
`context_attn_llama_repaired.py`. The source SHA256 is
`8317f23a4c8c04305b230369c2f59f6c4aaf41a5e80dca5336e88a1e13617e0b`.
All 96 original/repaired baseline files are preserved. The existing Host
results remain available in their original packages.

The entries are `Contract.lean`, `Kernels.lean`, `Correct.lean`, the auxiliary
`Misc/NativeCorrect.lean`, and the complete-pair `FPEquiv.lean` entry, under namespace
`VeriTile.Bench.Repairs.ContextAttnLlama.KernelOnly`.

## Input and launch domain

The output is the supplied ordinary CUDA FP32 Tensor and Python returns None.
Q/Out have shape `[T,H,D]`; K/V have shape `[C,Hkv,D]`. Payloads have canonical
contiguous strides, `D` is 16/32/64/128/256, and positive heads satisfy integral
GQA. Four contiguous INT32 vectors provide start, total sequence, prompt length
and request index. A canonical contiguous INT32 `[R,S]` table provides cache
pages. All views have resolved negative/conjugate flags. Nonempty data pointers
are four-byte aligned and complete storage spans satisfy the unchanged guard.

The common original byte heap allows every read role to alias any other read
role, including payload/metadata/page backing aliases. Only the complete output
backing is disjoint from all eight complete read backings. Nonzero storage
offsets are supported. Nominal source role names are fixed inside Request.call;
physical read separation is never a premise.

Metadata checks give `0 <= prompt <= sequence <= S`, valid requests and every
selected prefix page, query capacity, and disjoint positive query intervals.
Requested max is the unchanged nonnegative signed-INT32 bound; actual maximum
uncached length supplies the complete grid `(ceil(actualMax/BM), B*H, 1)`.
Empty calls skip the launch. BM=BN is 64 for the source's Tesla branch and 128
otherwise. Four warps are used through D64, eight for D128/D256; stages=1.

## What the proofs conclude

`TensorBridge.correct` constructs an actual successful Real-erased shared-source grid and
observes every valid logical output through the supplied Tensor's physical
address. Its target is an independent finite sum using gathered pages,
`exp(log 2 * score)`, the finite causal sentinel `-1e8`, masked zero loads, and
the padded key denominator lanes. It retains the five exact FP32 launch-scale
values obtained from the ordered Python expression
`1.0 / (D ** 0.5) * 1.4426950408889634`. The unrestricted Real observation does
not encode IEEE words or require payload finiteness.

This is the source's finite-sentinel formula. It is not ideal strict-causal
softmax: the historical control exposing a row-zero value of `5/63` is retained.
No numerical experiment is used to close a mathematical or program gap.

`TensorBridge.fp_equiv` constructs successful actual `FP.Structural.exec` runs for the
complete grid, connects them to an independently defined online recurrence,
and reconstructs all valid logical output FP32 words from the executed byte
stores. The selected SM90 model includes fixed-K8 TF32 WGMMA, physical ordered
replicas and routing, the stored row/channel owner, FMA, exp2, division and
casts. QK's first instruction clears scale-d; PV's first instruction retains a
live accumulator, including when its bits are zero. BM64 with D128/D256 uses
the corrected physical width/grouping plan.

Its only numerical laws are pointwise correspondences for eight scalar
instruction families and one fixed-K8 single-output WGMMA family. There is no
whole dot, reduction, graph, compiler, execution, output-equivalence, allocator
or Host-success premise. No native-to-Real equality is asserted.

Both Tensor application results include the exact output-byte complement, all eight full
untyped read-backing frames, and empty-call identity. Actual ordinary Torch
FP32 Tensor API laws, plus an explicit ordinary INT32 Tensor binding, connect
logical coordinates, storage offsets, pointer formation and the original heap.

`NativeSource.native_source` constructs every finite grid and source-loop
prefix and equates the actual quotation's operand evaluations to finite native
signed expressions. `NativeSource.source_connection` additionally connects the
actual prelude range header and finite loop control. `NativeSigned` retains
signed64 comparisons and the original min-after-prompt association.
`Safety.query_formed`, `all_formed` and `page_formed` derive signed64 element and
four-byte offsets for **all formed lanes**, including false-mask lanes and
inactive query programs. Metadata offsets and the final BM-step backedge are
also exact. No caller-supplied trace, reachability or coverage premise is used.

## Validation and boundaries

The public Audit checks standard axioms and noncircular contracts, independent
references, primitive laws and observers. Task-local extraction avoids the
historical repaired FPSuccess/current-Algebra ABI collision; the failed import
probe is retained. Pure original Real proofs are reused. No shared runtime,
compiler, cache policy or mathematical package is expanded by this revision.

The existing source/guard controls pass. Nine additional full CPU-interpreter
executions cover both tiles, all widths, lengths 63/64/65/127/128/129, prompts,
empty calls, nonzero offsets, full read/output-complement frames, all-read
backing aliases, and exceptional FP32 words. Finite cases compare an independent
gathered finite-sentinel formula. The exceptional case checks admission and
frames; it does not claim Real agreement.

Ten actual Triton specializations compile offline for SM90: all five D values
and both tiles, with ordinary unit-stride specialization, 16-byte-aligned and
four-byte-offset pointer classes. TTIR/TTGIR/LLVM/PTX are retained. These checks
are not GPU launches, native numerical conformance experiments, or certification
of the complete physical lowering plan.

Compiler realization of the selected native model, byte transport, ordinary API
behavior, and race-free device/resource delivery remain explicit external
boundaries. The shared launcher grid-product issue remains separately tracked;
it is not added as a kernel proof premise. The parent owns ledger/Lake and
canonical selection. Frozen receipts and dependency hashes are in
`.veritile-cache/kernel-evidence/context_attn_llama/delivery.json`.

## Public kernel IO

`context_attn_llama_correctness` now states
`Spec.Real (RealInterface.programIO.ImplementsOn Metadata formula)`. `RealIO.lean`
exposes eight finite typed input arrays, output masks/strides, the complete
ordinary source grid, original geometry and the independent finite-sentinel
formula. `Proofs/RealImplementation.lean` connects actual source access safety,
success, every active output and the exact exterior frame. Every previously
admitted raw state supplies this finite input binding and metadata; the original
canonical flattened physical transport remains in `Misc/KernelCompatibility.lean`.

`context_attn_llama_native` states the shared `ArrayProgramIO.ObservesOn`
judgment. `NativeIO.lean` and `NativeArrays.lean` expose the exact FP32 source,
all supplied jobs (including repetitions and an empty order), named typed
arrays and finite gather footprints. Arbitrary initial cell tags and native
expressions remain admitted. A pure array scatter updates every aliased role,
so later jobs see earlier writes; only start/sequence/prompt must differ from
output, exactly as before. Signed request/page values retain their original
`toNat` address conversion. The output initial-value view imposes no source
read footprint. Conservative key windows may include unused tokens of an
inactive query block; source success and result do not depend on capacity
premises. The native observer retains every consumer/replica. Only the existing
pointwise `ScalarLaws` primitive boundary is present; there is no optimization
rewrite, new numerical assumption, or native-to-Real claim.

Operational connections live in `Proofs/`: actual prelude, full range and
epilogue safety; finite-window coverage; arbitrary-order invocation execution;
independent array/reference agreement; output tags and exact cell frames.
`kernel_real` and `kernel_native` retain their original full statements and names
in `Misc/KernelCompatibility.lean`. The unchanged `TensorBridge.correct`,
`fp_equiv_lanes` and `fp_equiv` continue to derive physical observations from
the same established `Host.whole_call`/native-byte execution foundation.
`Correct.lean` retains its independent Real specification. The full native
statement and proof of `context_attn_llama_native` are preserved as an ordinary
theorem in `Misc/NativeCorrect.lean`; neither imports TensorBridge.
The optional `NativeInterface.native_observes` helper is in
`Misc/NativeObservation.lean`; `Proofs/NativeImplementation.lean` retains
the main `implements` connection.
`FPEquiv.block_end_factored_equiv` now proves complete actual programs
`original ≡[[]] blockEndFactored` on the original unrestricted array domain.
`Kernels.lean` directly quotes the complete selected FP32 original and candidate.
The candidate factors the shared prompt offset out of integer `minimum`, saving
one scalar add per CTA; no performance measurement is claimed.
`SourcePair.original_preserved` proves equality with the full historical source,
and `same_run/same_safe` prove exact complete-state/safety equivalence.

`PairIO` retains the original nine typed views (including initial output),
paged loads, runtime metadata, finite access windows, masks and any ordered job
list. Repeated jobs, output collisions and Q/K/V/request/page-output aliases
remain allowed. Only start/sequence/prompt metadata remain separated from output,
as before. The main pair concludes both safe successful runs, all typed final
outputs and both exterior frames. All native consumers are preserved by exact
carrier equality. No finite-word guard, floating rewrite or experimental rule
is introduced. The existing scale-first candidate and optional native
observations remain historical independent results. The separate Real weighted
sum, complete ordinary grid and original geometry/metadata domain are retained.

The 2026-10-20 IO-format pass performs targeted Lean builds and source/import
checks only. It does not rerun historical CPU controls, independent native
replay, GPU execution or numerical experiments.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.
The retained raw compatibility applications are also under `Misc/`; the main
readers do not import them merely to re-export their optional conclusions.
