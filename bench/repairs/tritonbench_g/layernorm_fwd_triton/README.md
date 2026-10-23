# Three-pass strided FP32 layer normalization: kernel proof

The current entries prove the repaired kernel itself. `Source.launch` executes
the canonical typed DSL for every program in the original `(S,H)` grid. The
public IO results prove actual source access safety, successful execution,
every output, and the complete memory frame. `Correctness.layernormIO` and
the retained optional `Equivalence.layernormIO` display the [S,H,N] channels, actual strides, complete
(S,H) launch grid and runtime epsilon register binding. Allocation, device/stream management, compilation, launch
availability, and wrapper return are outside this kernel-only scope. Earlier
wrapper documentation is retained in `HISTORY.md`.

The production repair and its guards are unchanged (SHA256
`a22618b08165692510fbf836cd2f7ad87943610902d3ee85016824e34e82aae9`).
The original source is also unchanged. The implementation uses three streaming
passes with block128, actual X/W/Y inner strides, pre-operation int64 widening,
and the source order `1 / sqrt(variance + eps)`.

## Independent target and exact domain

For each logical sequence/head row, the mathematical target is

```
mean = sum_col(X) / N
variance = sum_col((X - mean)^2) / N
Y = ((X - mean) / sqrt(variance + eps)) * W
```

`Contract.Math` defines these full-axis formulas independently of source state,
execution and native decoding. The Real theorem quantifies arbitrary physical
Real input values; it is an ideal interpretation, not an FP accuracy claim.
Nonnegative variance and positive inverse standard deviation are proved for
positive epsilon. Empty axes have no output coordinates.

The exact existing Python domain remains: strided FP32 CUDA X of shape
`[S,H,N]`, strided FP32 W of shape `[H,N]` on the same device, resolved negative
view flags, four-byte aligned nonempty data pointers, and a finite positive
Python int/float epsilon whose finiteness query succeeds. Python bool retains
its int behavior. All FP32 words, including NaNs, infinities, signed zeros and
subnormals, are admitted. There is no positive-dimension, contiguous,
canonical-stride, zero-offset, no-alias, or artificial small-shape guard.

Original `VeriTile.Torch.FP32.Tensor` objects, storage identities, offsets,
logical physical-index functions and actual `BitVec 64` data pointers enter the
public theorems. `View3`/`View2` state ordinary signed metadata and actual stride
interpretation laws. `TensorLaws` supplies storage bounds and pointer meaning.
The supplied output buffer has its actual strides, capacity, injectivity and
freshness; the proof does not execute an allocator. Read-only X/W aliasing is
unrestricted. Empty tensors need no unused alignment or address bound.

## Execution, native precision, and addresses

`SourceModel`, `SourceAccumulate`, `SourceOutput`, `SourceComplete`, and
`SourceContract` adapt the historical typed proofs to the repaired strides.
Only active output addresses need to be injective, preserving zero and huge
unused singleton strides. `KernelGrid` composes all programs and proves every
regional output and exterior cell, retaining exact dtype tags.

`KernelGraph` separately constructs the ordered input-only three-pass graph.
The graph preserves each masked load, centered-tail selection, chunk carry,
reduction, cast, and output operation. `KernelBoundary` derives its original
logical leaves from actual Tensor pointer reads; no output equation is an
input-binding premise.

The selected precision is FP32 X/W, loads, accumulators, statistics and output.
`KernelNative` executes source terms and observes them through individual native
instruction ports. Compiler profiles inspect expression shapes and physical
leaf identities, never raw input words. Adjacent multiply/add contraction uses
a genuine ternary `fmaRN`; operand order, literal-zero and unit-division
elimination, and identity FP32-cast lowering remain explicit. Both reductions
expand complete ordered trees; no callback can supply a reduction or final
result. `sourceOrderProfile` witnesses consistency of the profile interface.

`KernelABI` retains actual epsilon host/packed scalar kinds and their local
FP32 conversion. The converted word enters `eps_arg` for every program and
the independent output graph through the same external scalar leaf. Compiler
shapes erase its payload. Both public interpretations use one runtime source
quotation; the Real theorem supplies the ideal Real epsilon scalar. It proves modulo-64 products and pointer updates, including unused
and final increments. Active pointers equal the original storage addresses.
Output capacity bounds every executed column/lane addition and final loop
increment; all active FP32 addresses are aligned.

`KernelPhysical` interprets actual four-byte little-endian input reads and
ordinary FP32 stores at the supplied output pointer. Its per-store refinement
proves exact output bytes, the complete exterior byte frame including strided
holes, and preservation of every byte of both input backing storages.

`Correctness.layernorm_fwd_triton_kernel_real` uses shared `ProgramIO.Implements`
and an independent Real formula. `Equivalence.layernorm_fwd_triton_kernel_native`
uses shared `ProgramIO.Observes`: input values remain arbitrary symbolic terms,
and the pure native observer maps output terms to FP32 words. No new input-leaf
restriction is introduced. The retained native theorem remains optional; the main FP entry now compares two actual schedules.

The optional complete Tensor applications are:

- `TensorBridge.layernorm_fwd_triton_real`: typed full-grid success, every
  independent logical Real output, and the full regional frame.
- `TensorBridge.layernorm_fwd_triton_native`: typed full-grid success, every
  observed native output with the actual converted epsilon argument, full
  regional frame, and actual native pointer facts.
- `TensorBridge.layernorm_fwd_triton_native_physical`: actual output bytes,
  exterior/input backing byte frames, and native output-byte addresses.

`TensorBridge` also exposes exact Python-check/precondition equivalence. Its
premises contain no historical host service, successful-wrapper witness,
`LocalConformance`, numerical transformation identity, or split/FMA equality.

## Boundaries and validation

Primitive arithmetic, division/sqrt approximation and flush behavior, casts,
scalar ABI conversion, compiler choices, and ordinary Tensor/storage semantics
are explicit modeling boundaries. No native GPU numerical conformance or
FP-to-Real error bound is claimed. There are no deferred numerical transformation
obligations or numerical experiments in this kernel revision. Availability of
the original logical grid on a specific runtime remains outside kernel scope.

The original `[1,2,2]` input with strides `[4,1,2]` writes physical cell1 twice
and leaves cell3 unwritten. The current source regression reproduces that
collision and verifies the repaired scatter. The Lean counterexample proves
the colliding original and distinct repaired address expressions.

Targeted CPU source regression passes 13 shape/view cases, 16 rejected calls,
bool epsilon, and the original collision. Cases include all empty axes,
singletons, tails, multiple chunks, offsets, broadcasts, input aliases and huge
unused singleton strides. CUDA and host operations are mocked; these are source
and storage checks, not native numerical conformance. Offline lowering checks
cover the retained five SM80 integer/specialization profiles separately.

Current build receipts, standard-base axiom audits, source hashes and validation
logs are recorded in the frozen delivery packet after final review.

The primary `layernorm_fwd_triton_kernel_real` and optional
`layernorm_fwd_triton_kernel_native` in `Misc/NativeCorrect.lean` take shape/strides,
output-coordinate injectivity, and scalar operands; shared IO binds logical
inputs and keeps execution states out of the main statement. Complete Tensor
applications remain unchanged in `Misc/TensorBridge.lean`; `KernelBoundary` supplies
the original-input correspondence to the same `KernelGrid.complete` foundation.

`Proofs/SourceSafetySteps.lean` checks all three actual loop bodies, and
`Proofs/SourceSafety.lean` composes their reached states using the existing
accumulator invariants. `Proofs/PublicIO.lean` composes the entire row grid and
logical input/output/frame connection. The previous raw operational results
remain ordinary theorems in `Proofs/KernelRealResult.lean` and
`Proofs/KernelNativeResult.lean`.


The main `FPEquiv.source_equivalence` now compares the actual original and
candidate complete grids. `layernorm_fwd_triton_scheduled.py` moves each output
weight load after `x_hat`, shortening the weight vector's lifetime. This is a
register-scheduling candidate; no measured speedup is claimed. The three streaming
passes, every loop iteration, reduction order, FP32 operation/cast, mask, stride,
int64 Python index widening and output remain unchanged.

`Kernels.lean` contains ordinary complete `triton` quotations for the historical
literal instance, actual runtime instance, and candidate. Runtime epsilon uses
standard `tl.param eps_arg : tl.float32`, with the original per-invocation scalar
binding and no dummy assignment. Complete original and runtime AST identities
against the former frontend/template are checked by reflexivity in the campaign
preservation artifact. Proof-only source decomposition lives in
`Proofs/SourceModel.lean`.

`PairSource` proves the scheduled body returns the same complete state by
commuting independent register assignments. `PairExecution` lifts this through
all three loops and proves candidate access safety at every reached state.
`PairIO` covers the complete `(S,H)` grid, both successful executions, every
output and both full exterior frames. The existing output-coordinate injectivity
condition remains explicit; read-only X/W aliasing, arbitrary accepted strides,
empty axes, tails and all input words remain admitted. The main pair has an empty
experimental rule set and introduces no numerical experiment obligations.

The candidate passes the existing bounded CPU checker: 13 full source/storage
cases, 16 guard rejections, bool epsilon and the original inner-stride collision.
The complete Python AST differs only by public identifiers and the documented
load scheduling. The independent Real theorem and optional native/Tensor results
remain at their original scope.

可选证明入口统一位于 `Misc/`：`Misc/TensorBridge.lean`, `Misc/NativeCorrect.lean`, `Misc/PublicAudit.lean`。文件迁移保留声明全名、类型、证明和原有状态；主入口不再为兼容重导出可选 native 结果。所需 IO、公式及连接助手继续保留原位置。
