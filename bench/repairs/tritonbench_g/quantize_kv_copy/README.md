# Grouped quantized destination scatter

`Correct.lean` exposes the shared `ProgramIO.ImplementsOn` specification;
`Misc/NativeCorrect.lean` retains the ordinary native theorem for the complete Real/native source, both output channels and the
exact exterior frame. The public layout and actual launch schedule are in `IO.lean`.
The full canonical mixed-dtype Tensor and common-memory applications live in
[Misc/TensorBridge.lean](Misc/TensorBridge.lean), together with the checker, historical
checked-wrapper/local-transport and Tensor-pointer theorems. Their statements
and numerical conditions are preserved.

`destindex_copy_quantize_kv` writes INT8 codes and FP32 scales into the caller's
two output tensors and returns `None`. Each head is split into groups of eight.
The original `K.view(rows, heads, groups, 8)` and equivalent output view remain;
they share the caller buffers. The original two-dimensional `(tokens, heads)`
launch is preserved. No payload copy or snapshot is introduced.

The accepted repaired JIT is unchanged: it restores the input/output lane strides,
scale group stride and destination stride, widens controls to INT64, and assigns
zero groups scale zero and divisor one. The direct contract rejects strided views,
but the repaired source still records all four strides faithfully. Input and scale
production formats are FP32, codes INT8, and metadata INT32.

All four arguments must be contiguous strided tensors of rank 3/1/3/3 on one CUDA
device, with no unresolved negative view. Heads are positive, feature width is a
positive multiple of eight, and source rows cover the destination count. Output
head/feature dimensions match the input and scale shape is
`[capacity, heads, feature_width/8]`. Destination rows are bounded and unique.
The four direct data spans are pairwise disjoint and naturally aligned. Every
extent is below `2^63`, token count below `2^31`, head count below `2^16`, and the
padded group/eight-lane tile has at most `2^20` elements. Empty destination lists
are admitted. `checkArguments_iff` proves the complete ordered rejection model
matches the typed theorem contract, with a separate non-Tensor classification.

The source-only `Kernels.lean` contains the typed source event graph and the
mutable-byte memory model. Native proofs establish exact signed masks, actual
2D IDs, signed INT32 metadata widening, full INT64 pointer SSA, nonwrapping accesses,
view address equations and launch-coordinate bijection. `Source.program_exact`
and `Source.grid_exact` connect every finite source event to the complete mutable
grid. The direct wrapper result identifies the original output objects and
metadata, all code/scale bytes, and every exterior backing byte. Input and metadata
reads are proved unchanged throughout execution; program store footprints are
disjoint. No structural premise is deferred.

`Correct.lean` is an independent mathematical result over arbitrary Real input and
scale cells. It proves the logical group-of-eight maximum, division and toward-zero
quantizer, including zero groups, without finite-byte decoding. Its target has no
execution/source-register dependency. `Misc/NativeCorrect.lean` independently proves the
unconditional source-order structural result for every admitted FP32 bit pattern.
The five native instruction ports and the eight-element reduction tree remain
explicit primitive-model boundaries. Optional local max-node, division, absolute-
value, scale cast/promotion, zero-test and INT8-conversion correspondence hypotheses
remain in the retained Tensor/legacy transport theorems. They are not new per-task
numerical experiment obligations. A separate theorem gives signed toward-zero byte
meaning on the defined conversion trace;
nonfinite or undefined conversions do not become hidden input restrictions.
Software zero behavior and FP32 scale-byte encoding are proved directly.

Validation: task modules compile with clean standard-base axiom audits. Run
`python bench/repairs/tritonbench_g/quantize_kv_copy/check_contract.py` for 9 admissions,
52 rejections, original-view storage checks, exact output/exterior-byte checks and
four original stride-defect witnesses. The accepted JIT AST is pinned and executed
on CPU backing storage. CUDA context and launches are mocked; no GPU numerical
conformance is claimed.

The reader interfaces expose `Correct.mathematical` and `FPEquiv.native` (retained in `Misc/NativeCorrect.lean`) through
`IO.lean`'s FP32 input, signed destination metadata, INT8 code and FP32 scale
channels. The destination-channel condition is the original signed read binding;
store uniqueness and group-width coverage remain explicit theorem hypotheses.
Source and result scopes retain every token/head program and all groups of eight.

`Proofs/AccessSafety.lean` checks the actual source trace, including the unmasked
destination load and both masked output stores. `Proofs/IOGrid.lean` proves exact
correspondence between the fixed shared launch executor and `KernelGrid.run`,
keeps input/metadata reads stable across the grid, and composes the existing
independent outputs with the shared IO frame. `KernelRaw.mathematical` and
`KernelRaw.native` retain the earlier complete raw statements; no source
execution proof is duplicated. Supporting modules import no reader entry.

Complete mixed-dtype Tensor/byteheap endpoints remain ordinary `TensorBridge`
theorems with their original contract, observations and empty behavior. The
selected Python, canonical kernel source, Contract and numerical boundaries are
unchanged. No GPU execution or independent numerical replay was added.

## Implementation pair (2026-10-21)

`FPEquiv.address_early_equiv` compares the complete original source and
`KernelSyntax.addressEarly`, through the shared `ProgramIO` pair judgment.
The candidate schedules the existing `o_ptrs` and `os_ptrs` pure assignments
before `abs_data`, exposing address formation alongside the FP32 quantization
pipeline. No performance measurement is claimed. The companion Python candidate
is `quantize_kv_copy_address_early.py`; its complete module AST differs only by
this assignment scheduling.

All floating operations, the eight-lane reduction tree, casts, INT64 controls,
load/store masks and two stores retain their source order and precision. The
proof establishes complete state equality and trace-safety equivalence, lifts
these to the entire token/head launch schedule, and reuses the existing native
source theorem for success, both outputs and the exterior frame. Every FP32
word is admitted, including nonfinite values under the explicit total byte
conversion profile. Group coverage and injective target metadata are unchanged.
The experimental rule set is empty: no floating numerical rewrite is used.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/NativeCorrect.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
