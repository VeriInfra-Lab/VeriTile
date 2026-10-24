# Quantized destination scatter

The selected source is `quantize_copy_kv_repaired.py`, SHA256
`6deb9bd29829b591b52cf53d63da2288804b71cc9c4833e834ff048e72909ccc`.
This revision changes neither that Python source nor its guards. It keeps FP32
input, INT64 destination metadata and offsets, INT8 codes, and FP16 scales.
For each source token and head, the kernel reduces `abs(input)`, divides by 127,
rounds the scale to FP16, promotes that scale for the FP32 quotient, and scatters
the code and scale to `DestLoc[token]`. An original maximum of zero selects
denominator one and preserves the zero scale representation.

The production contract admits contiguous strided tensors on one CUDA device,
with ranks 3/1/3/3 and no unresolved negative views. Heads are positive; feature
width is a positive power of two. Shapes agree; scales have shape
`[capacity, heads, 1]`; destinations are bounded and unique. Data spans are
disjoint and naturally aligned. Existing program-count, INT64 extent and padded
tile bounds remain. Empty destination lists are admitted. There is no finite
payload restriction: NaNs, infinities and nonzero rows whose FP16 scale becomes
zero remain admitted. `Contract.lean` and `Proofs/Contract.lean` retain the exact
ordered Python rejection model. `TensorBinding` supplies the relationship to
actual allocations and the initial metadata bytes, rather than another payload
guard.

`Kernels.lean` now holds the sole readable Triton DSL source. The second version
only commutes `1 * offs_d` to `offs_d * 1` in an input address. All floating
operations, their order, masks and casts are unchanged. No separate optimized
Python implementation or performance improvement is claimed.
`Proofs/KernelSource.lean` is a compatibility import. The previous scalar/event
model has moved to `Proofs/Model.lean`; its existing proofs are retained.

`IO.lean` displays the signed metadata channel, FP32 input, INT8 code and FP16
scale windows, and both full token schedules. The metadata condition binds the
actual signed input values to the same natural destinations used by those
windows. `Proofs/AccessSafety.lean` checks every actual metadata/data load and
code/scale store for both source versions. `Proofs/Interface.lean` carries the
same input bindings through every source execution state and composes the
existing complete output/frame proof. `Proofs/RawReal.lean` and
`Proofs/RawNative.lean` retain the former raw statements; the reader entries now
expose independent logical formulas and shared `ProgramIO` relations.

`Correct.kernel_correct` in `Correct.lean` proves successful complete launches
of both shared kernels, all destination code and scale cells, and every exterior
logical memory cell. Its independent Real target is

```text
m = max_j |input[token, head, j]|
scale = m / 127
code = 0                                      if m = 0
       signedByte(trunc(input[column]/scale)) otherwise
```

`Proofs/KernelRealModel.lean` interprets that same `ComputeKernel` with Real
arithmetic, a genuine nonempty mathematical maximum, erased FP16 rounding and
explicit mathematical toward-zero signed-byte conversion. It does not claim
that the generic mathematical interpreter already supports that conversion, or
that native arithmetic exactly equals Real arithmetic. Logical Real inputs are
arbitrary; they are not constrained to a finite byte codec. This public theorem
is about the kernel, not wrapper execution.

`FPEquiv.kernel_equiv` in `FPEquiv.lean` states
`original a rest p count dest ≡[[]] indexCommuted a rest p count dest`.
`Programs.lean` binds these views to the two existing complete token schedules
and their metadata domain. Both schedules are proved safe and successful,
their exact typed output cells agree, and every exterior cell is framed.
The empty experimental rule set uses the proved integer-address rewrite;
all native arithmetic and conversion ports retain the same model on both sides.
`NativeCorrect.kernel_native` separately retains the ordinary single-source
implementation theorem for the independent native code/scale target.
`Proofs/ProgramEquivalence.lean` reuses the existing complete schedule proof.
The stronger existing
common-final-state witness remains in `Proofs/RawNative.lean`. `TensorBridge.kernel_equiv`
retains the complete physical-input and native-store application; its `NativeReads`
contains only initial read equations, never a desired output or execution premise.
`TensorBridge.native_pointer_binding` and `TensorBridge.native_metadata_binding` reuse the checked INT64
arithmetic to identify the actual input/output byte addresses, including both
bytes of each FP16 scale and the destination metadata read.
`TensorBridge.native_signed_metadata_binding` identifies the typed metadata read with the
signed INT64 interpretation of the same eight physical bytes.

The native arithmetic interface explicitly retains `abs`, `div.full.f32`,
FP16 rounding and promotion, equality, and the full reduction shape, axis and
order. Equivalence is parametric in those shared native callbacks: it needs no
IEEE-division replacement, reassociation, reduction permutation or primitive
agreement assumption. Basic instruction semantics are not deferred numerical
experiments. The signed conversion profile in `Proofs/PtxSigned.lean` is total
for the selected emitted sequence `cvt.rzi.s16.f32; st.global.b8`: truncate toward
zero, clamp to signed 16 bits, then observe the low byte. Its independent
conversion specification, uniqueness, bounds, all-NaN result and overlap with
the existing partial INT8 converter are proved. In this profile NaN stores byte
0, positive infinity stores byte 255, and negative infinity stores byte 0.

This native instruction profile has an explicit compiler boundary. Offline
compilation of the unchanged Python with Triton 3.7.1, CUDA sm80, one warp and
one stage produced `abs.f32`, `max.f32`, `div.full.f32`, `cvt.rn.f16.f32`,
`cvt.f32.f16`, `cvt.rzi.s16.f32`, and byte/halfword stores, without `.ftz`.
The saved configurations are heads/width 1/2, 3/4 and 3/64. The offline
`ASTSource` signatures contain four pointers and pin wrapper stride/head-count
values as constexpr; they are explicit source specializations, not a trace of
the actual runtime JIT binder. They do not prove identical lowering for every
runtime signature, specialization or future compiler. LLVM `fptosi`
outside its representable truncated range produces poison; the selected PTX
profile does not assign that poison a unique portable meaning.

`Proofs/KernelCounterexamples.lean` proves that the historical software model's
defined-conversion premise does not follow even from finite inputs: FP32
`2^-20` gives an FP16 zero scale, an undefined partial signed conversion, and an
observable dependence on an arbitrary extension of that partial converter.
These are exact Lean computations about that old model, not GPU observations
or a newly added input restriction. The old conditional results remain in
`Proofs/LegacyCorrect.lean` and `Proofs/LegacyFPEquiv.lean`, with their original
names and hypotheses. Their primitive-agreement hypotheses are historical
conditional model statements, not pending numerical experiments for the new
public kernel theorem.

The public scope is typed logical output/frame, initial physical reads, exact
native addresses and final INT8/FP16 store observations. `Proofs/KernelStores.lean`
connects the final code cell to its one-byte observation and the scale cell to
both bytes of the actual FP16 result, at the same native Tensor addresses.
Distinct caller spans and unique destinations are explicit contract conditions.
The complete typed frame covers input/metadata regions, untouched destination
rows and every other region. The retained legacy event model additionally has
physical-byte output/frame proofs; this revision does not introduce a separate
whole byte-heap interpreter or claim a composed after-heap execution theorem.
That scope boundary is not a numerical deferral. Whole-task acceptance is
reviewed against this explicit public target; this task does not edit the ledger.

Validation uses the task-local closure build and embedded standard-base axiom
and non-circular-specification audits. `check_contract.py` passed the unchanged
accepted JIT AST, 9 admissions, 51 rejections, code/scale and exterior-byte checks,
object identity, and the original strided-metadata counterexample. That replay
uses CPU backing storage with mocked CUDA context and launch. No GPU execution
or GPU numerical conformance result is claimed.

Evidence is kept in
`.veritile-cache/kernel-evidence/quantize_copy_kv/`: `TARGET.md`,
`native-probe.json`, the corresponding TTIR/TTGIR/LLVM/PTX artifacts,
`final-targeted-build.log`, and `final-cpu-replay.log`. Primary references are
the official [PTX conversion rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt),
[PTX store rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st),
[PTX division rules](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div),
and [LLVM fptosi rules](https://llvm.org/docs/LangRef.html#fptosi-to-instruction).
The saved official documents and shared PTX extracts are referenced from the
sibling `kv_cache_filling` evidence packet without duplicating them; the local
`fptosi-to-instruction.txt` records the signed-conversion LLVM boundary.

`Correct.lean` exposes the independent mathematical specification; `FPEquiv.lean` proves the actual `original ≡[[]] indexCommuted` program pair. `Misc/NativeCorrect.lean` retains the single-source native theorem. `TensorBridge.kernel_correct` and `TensorBridge.kernel_equiv` retain the complete Tensor applications, alongside pointer and metadata binding aliases.

The mathematical specification and program-pair proof use raw geometry, destination indices and logical IO bindings. Their supporting proofs retain both complete launches, every code/scale output and the full exterior frame. `TensorBridge` retains the complete Tensor applications and physical store observations through the same raw grid proof and existing input/address correspondence.


可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。

The 2026-10-21 source-only cleanup writes both retained versions as complete
ordinary `triton` blocks directly in `Kernels.kernel`, removing the former
whole-source macro. The complete function is definitionally equal to the frozen
prior source for both versions and every parameter, checked by `funext`, cases
and `rfl`. Signed metadata, FP32 input/arithmetic, zero-maximum handling, FP16
scale, int8 conversion, masks, stores, public IO and the existing Real/main pair
statements are unchanged. This cleanup does not alter proof completion or add
numerical, native, host or hardware validation requirements.
