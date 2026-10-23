# Direct FP32 block quantization contract

`FPEquiv.bias_fold_equiv` compares the original full source with
`int8_quantization_bias_fold.py`. The candidate replaces
`0.5 * where(sign, 1.0, -1.0)` with `where(sign, 0.5, -0.5)`, removing
one per-lane multiplication before the INT8 conversion. All classification,
reduction, division, clipping, source branch and store operations remain in
order. This is an operation-count optimization; no speedup has been measured.

`Kernels.lean` exposes both complete ordinary `triton` programs. The pair
uses the fixed FP32 software multiplication profile and formally checks the
two constant products `0.5 * 1.0 = 0.5` and `0.5 * -1.0 = -0.5`.
Other source instructions remain explicit primitive ports. The theorem
admits every FP32 bit pattern, including NaNs, infinities, signed zero and
subnormals; it adds no finite-value guard. Both complete launch schedules
succeed safely and return equal INT8 and FP32 cells with exact exterior
frames. The experimental rule set is empty. The existing independent total
Real specification and optional Tensor/native results remain available.
`check_bias_fold.py` checks that the complete Python modules differ only in
the stated constant fold and runs the existing complete CPU source harness
against the actual candidate.

This revision keeps the accepted revision002 kernel and passes each original
input directly to it. Q and K independently have rank 3 or 4, independent shapes,
CUDA devices, and positive integer block sizes. They may share storage. The
wrapper returns `(q_int8, q_scale, k_int8, k_scale)` with each int8 tensor retaining
its corresponding input shape and device. Each FP32 scale tensor has shape
`(*input.shape[:-2], ceil(length/block), 1)` on that input's device.

Inputs must be strided, contiguous FP32 CUDA tensors with no unresolved negative
view bit. Nonempty inputs need 4-byte data-pointer alignment. A positive integer
block argument excludes Python bool; `block * channels <= 2**20`; input element
count and scale element count are both below `2**61`. The wrapper checks all of
these before allocating that input's outputs. Storage offsets are retained.
There is no shape/device equality requirement between Q and K and no finite-value
input guard. Empty batches/lengths return empty allocations. Zero-channel inputs
return empty int8 tensors and zero-filled FP32 scales.

Q multiplies by the production host constant `channels**-0.5 * 1.44269504`; K
skips this operation. The source widens program IDs and lanes to int64 and chunks
the flattened program grid at 2147483647. Each block checks transformed values,
sanitizes invalid values, reduces maximum absolute value, divides by 127, uses a
safe denominator for zero/underflowed scales, clamps to [-127,127], adds a signed
0.5 bias, and converts toward zero to signed int8. Invalid blocks produce zero
bytes and zero scales. Loads, arithmetic, and scales use FP32. Fusion is disabled.

The dynamic admission theorem is
`TensorBridge.python_checks_iff_preconditions`. `Request.contract` is exactly
that checked domain; ordinary Tensor metadata/storage laws derive the direct
pointer relation. `Direct.native_pointers` derives native masks and all widened
input/output addresses from the checked contract and ordinary allocation bounds.
The separate scale-pointer theorem proves FP32 byte scaling. Every logical
batch/row/channel is covered by `TensorBridge.all_coordinates`; every scale is
covered by `Schedule.complete`, including launch-chunk boundaries.

The optional `FloatingSpecification.per_block_int8_native` theorem is retained
in `Misc/NativeCorrect.lean`; it is a single-source native result.
It uses the shared `ProgramIO.Implements` judgment on `IO.lean`'s complete
source schedule and independent logical native graph, with scalar instructions
and an ordered reduction plan explicit. The full previous raw results remain
in `Proofs/NativeRaw.lean`; `QuantizationIO.software_implements` in
`Proofs/NativeInterface.lean` retains the separate conditional software relation
over logical inputs with exactly the existing context and fourteen primitive
ports. The native result proves those agreements and admits all words.

`TensorBridge.per_block_int8_ordered_graph` retains the complete Tensor application.
It proves successful source launches for Q and K, exact returned objects,
shapes/devices, every INT8 byte and FP32 scale, the live caller frame, and every
byte outside the four output allocations. Its independent reference is
`NativeGraph.target`, evaluated on both original input memories, including
aliases. `NativeGraph.Instructions` contains only scalar instruction ports;
the maximum expands an explicit ordered `ReductionPlan` whose validity includes
every lane exactly once and records padding. No whole-vector numerical callback,
source-result equality, or fourteen-operation agreement is assumed. The
existing `TensorBridge.per_block_int8_wrapper_fp32` and optional `per_block_int8_equiv_if_local`
transport theorem remain available as supporting interfaces.

`TensorBridge.original_tensor_words` connects the typed initial loads
to `Torch.FP32.logicalValue` using exact little-endian FP32 byte assembly.
`Direct.input_actual_pointer` connects the same storage view to the Tensor API
pointer checked by the guard. These are equalities for actual storage reads,
not a supplied relation to an unrelated logical array. Ordinary input-liveness
and fresh-allocation API laws protect aliases while K runs after Q.

`Specification.per_block_int8_real` is the primary mathematical kernel result.
It uses the shared `ProgramIO.Implements` judgment with an independent logical
total target. Its public parameters retain width/block positivity and the ideal
Real factor equation. Source execution, actual access safety, both output
channels and the exact frame are proved conclusions. No finite-input or
positive-channel restriction is added. The old raw theorem and complete logical
coverage theorem are preserved in `Proofs/RealRaw.lean`.

`TensorBridge.per_block_int8_real_complete` retains the complete Tensor application, combining the total real target with successful source launches and the full exterior-byte frame. `TensorBridge.per_block_int8_real_wrapper_total` remains its
value-only supporting theorem. This is a separate mathematical
interpretation of the same typed source, full grid, empty/zero-channel branches,
and returned four-tensor wrapper. It takes arbitrary real values at original
storage addresses, so it requires no encoding of arbitrary real inputs into FP32
words. Its independent piecewise target returns zero output bytes and scale for
any block with a transformed value outside FP32's finite range. Valid blocks use
block maximum absolute value, scale `max/127`,
and nearest-away rounding of `127 * transformed / max`, with zero normalization
when the maximum is zero. The retained `TensorBridge.per_block_int8_real_wrapper_correct`
corollary restricts transformed values to that range and uses the unbranched
max/127 target. This is not a Python guard. The total theorem covers arbitrary
mathematical inputs and the structural FP theorem covers all FP32 payloads.
The final FP32 scale storage conversion `store32` stays explicit; no claim
identifies all exact-real intermediates or scales with FP32 arithmetic.

Primitive model boundaries are production binary64 literal conversion and
power/multiply for Q's host constant; FP32 literal conversion, abs, multiply/divide/add and their
rounding/FTZ behavior; maxnum and ordered comparisons; and toward-zero signed
int8 conversion. `HostPrimitives` retains the binary64 literal conversions and both host
operations separately, before the final FP32 literal conversion.
The host-constant premise is required only for a nonempty launched Q input with
positive channels; empty and zero-channel routes never evaluate that constant.
FP32 memory loads/stores retain the exact four bytes. The upper clamp uses the
DSL's comparison/select interpretation with the fixed finite right operand
`+127`; no floating reassociation or arithmetic replacement is asserted.
The compiler-selected reduction tree is explicit structural data, not an
assumed equality of reductions. There are no deferred numerical transformation
experiments. The theorem is parameterized by scalar instruction semantics and
a reduction profile; it does not prove GPU hardware, the compiler, or PyTorch.
Successful allocation and race-free launches remain ordinary runtime boundaries.

`check_source_profile.py` offline-compiles the selected Q and K source on sm80
with `L=5,C=3,BLK=4,WIDTH=16,num_warps=4` and disabled fusion. The sampled profile
has FP32 loads/arithmetic/scales, ordered comparisons, maxnum/minnum reductions
and clamping, `mul.rn.f32`, `add.rn.f32`, two `div.full.f32` instructions,
`cvt.rzi.s16.f32` followed by an 8-bit store, and int64 address arithmetic.
The production compiler folds the signed half multiply to signed half literals
and can move a zero select across the clamp; these artifacts report compiler
lowering and are not a separate Lean proof of the compiler. Run the audit with
`python bench/repairs/tritonbench_g/int8_quantization/check_source_profile.py`.
It records source/compiler artifact hashes and performs no CUDA launch.

See `ORIGINAL_SOURCE_DEFECTS.md` for the independent formula and concrete original
shape, mask, zero-scale, stride, device and invalid-value counterexamples.

Historical provenance: accepted revision002 Python SHA256
`011d56e3dab83a0b4f6a9fc862628edd527ea3bfdac57dc1c8b4497587653c02`, reviewed by
`2026-10-18/int8-quantization-revision002-batch-integration/repair-review.json`
(SHA256 `1336dd13d226a7d8f74fd69c4656e3f5a346e86b0ba3511005bfc14c7172a2e1`).
The current wrapper adds the explicit direct-layout/extent/alignment contract and
removes input materialization; this provenance is not a hash of the new wrapper
or a change to the historical acceptance ledger. Original benchmark Python is
unchanged.

Run `python bench/repairs/tritonbench_g/int8_quantization/check_contract.py`.
It executes the actual repaired kernel/wrapper AST on CPU storage and checks 31
complete four-result cases, 15 guard rejections, independent shapes/devices,
aliased inputs, storage offsets, empty and zero-channel cases, NaN/Inf/overflow,
underflow, and launch chunk endpoints. CUDA allocation and launch are mocked.
Lean source/grid/native/result modules and public entries are checked with
targeted Lake builds and standard-base axiom audits. All proof segments are
selected from the single `Kernels.source.surfaceBody`; there is no second
manually maintained kernel AST.

`Misc/TensorBridge.lean` retains all original complete Tensor statements and their full domains, alongside wrapper corollaries and original-input bindings. `kernel_correspondence` connects the live wrapper state to the same `RawKernel.complete` result used by the main interfaces. Empty/no-launch and zero-channel scale initialization remain ordinary wrapper behavior; their existing observations and frames are retained. Host binary64 factor calculation and its FP32 literal conversion remain explicit in the bridge and native instruction models.


`IO.lean` names the FP32 source input window, INT8 output window and FP32 scale
window for every group in the existing schedule. Structural cells are typed
Real/Int/Real because that is the actual source interpreter representation;
storage and arithmetic precision remain in the unchanged source. The input and
output windows use the actual lane/row mask; every launched group writes its
scale, including a group with no active payload. There is no private scratch.
`Proofs/SourceSafety.lean` proves every actual load/store is backed, and
`Proofs/IOBindings.lean` connects the public windows to the existing full grid.
The Real/native interfaces reuse the old complete execution/value/frame
results; no second source, per-block substitute, numerical experiment or added
input narrowing is introduced. TensorBridge statements remain unchanged and
share the same `RawKernel.complete` execution foundation.

The optional `per_block_int8_equiv` theorem in `Misc/NativeCorrect.lean` retains the most general
source/software IO relation with arbitrary original primitive model and
software numerics. Its explicit logical domain retains the two per-group
context laws and fourteen reached local agreements, including empty-schedule
vacuity. `QuantizationIO.native_implements` is the ordinary native specialization
derived by proving those agreements; it is not a replacement for that domain.

## Selected pair review (2026-10-21)

Missing pair target: the software quantization formula is not a second source implementation.
The previous single-source native/software results now live in
`Misc/NativeCorrect.lean` as ordinary theorems. Their fully qualified names, complete
statements, proof bodies and input domains are retained when importing
`Misc/NativeCorrect.lean` explicitly. They are useful auxiliary correctness results and do
not count as a completed two-implementation FP equivalence. The independent
mathematical `Correct` target is unchanged.

Optional native observations, Tensor/host applications and reader/audit entries
are in `Misc/`. Import these modules explicitly when needed; the primary
`Correct.lean` and `FPEquiv.lean` entries do not re-export optional results.
