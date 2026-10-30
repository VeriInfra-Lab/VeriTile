# mul_exponent_compensator kernel proof

The canonical selected repair is `mul_exponent_compensator_repaired.py`, SHA256
`c944bb5a524eac38541b0e01b75de94a25a3e24bebaf2ff319d4c805aa7e15d1`.
The Python source, precision, guards and all 21 caller block sizes are unchanged.
The independent mathematical result is `x[i] * 2^112`.

`Kernels.candidateKernel` is the single readable FP32 DSL transcription of the
complete six-statement kernel. The constexpr preserves the original
`2.0 ** (127 - 15)` and the multiplication preserves `x * constant` operand order.
The DSL uses natural-number indices; `Native` lowers that same source to the
actual pre-arithmetic i32-to-i64 extension, i64 multiply/add/signed comparison,
and u64 byte addressing. That finite lowering consumes the same kernel’s
dtype-erased `.body`; the chosen scalar profile separately fixes FP32 arithmetic.
`KernelArithmetic.program` proves the actual single-program memory effect.
`Proofs/ScheduledSource.lean` reuses it to prove the complete grid for every
permutation, including in-place scaling; `Proofs/SourceSafety.lean` proves the
actual evaluated load/store masks are within the declared input/output windows.

`IO.lean` exposes one finite input and output channel, their contiguous windows,
all active masks, no private scratch, the actual typed source and the full
caller-selected grid order. Equal source/destination region names express
in-place scaling. The shared `ProgramIO.Implements` judgment includes input
binding, source access safety, successful execution, every typed logical output
and the complete exterior cell frame.

`Correct.kernel_real` has the independent formula `x[i] * 2^112` under `Spec.Real`.
The ordinary `FPEquiv.kernel_fp_profile` theorem in `Misc/NativeCorrect.lean` has the
independent exact `FP.mul profile x constant` target for every concrete FP32
profile and every input word. Only the Real result is a public specification.
The native result is a source-to-native observation, not an invented
optimization pair or an equality between native arithmetic and Real arithmetic.
The source IO covers every positive natural block size; the original finite
index and actual Tensor consequences retain all original block/grid guards.

`Proofs/PhysicalIO.transport` connects this shared source result to the original
finite lowered `Native.sourceLaunch`. It maps same-address input/output buffers
to one logical region and disjoint buffers to two, writes the actual source
outputs back to the original flat addresses, and proves the entire resulting
flat memory equals the finite execution using `source_launch_equal` and the
existing program-order/output/frame laws. Original signed-index, mask-tail,
allocation and alias conditions are retained. `Proofs/PhysicalCompatibility.lean`
preserves the three old raw statements as ordinary `kernel_real`,
`kernel_fp_profile` and selected-profile `kernel_fp` consequences, also exposed
by `TensorBridge`.

The full actual FP32 Tensor applications remain in optional `Misc/TensorBridge.lean`
as `TensorBridge.correct`, `TensorBridge.fp_equiv_profile` and `TensorBridge.fp_equiv`. Their inputs are already prepared input/output
buffers and ordinary Tensor API laws. They now use `Proofs/TensorIO.execute`,
which applies the shared source IO and exact finite-address transport.
Every logical output, every exterior
word, and the unchanged input are proved for every complete sequential program
order. This is not a new hardware-interleaving conformance theorem.
The floating theorem additionally exposes the exterior byte frame and uses the
shared Torch logical reader. Optional `Misc/TensorBridge.lean` exposes
`TensorBridge.actual_pointers`, `TensorBridge.actual_bytes` and
`TensorBridge.formed_indices` to connect the execution to observed data pointers, physical
Tensor indices, live storage bounds and the widened signed source arithmetic.
`TensorBridge.empty` applies the empty-grid identity to the Tensor buffers.
No new allocation, device selection or return-identity proof is required.
The source masked load has no `other` value. Inactive lanes never reach a store;
no result claim depends on their model placeholder or unspecified GPU values.

The checked input domain remains positive-rank, strided, contiguous FP32 CUDA,
without a negative view bit, and with a power-of-two block from 1 to `2**20`.
Default block 1 and Python bool subclass behavior are unchanged. Nonempty
pointer alignment, current-device and grid limits retain their existing checks.
Nonzero storage offsets and arbitrary singleton strides accepted by contiguity
remain supported. Empty tensors retain their original unused-offset behavior;
the zero-program grid returns the entry kernel memory unchanged. Masked tails
and every admitted block are covered without a new `numel < 2**31` restriction.

## Concrete numerical scope

`KernelArithmetic.constant_exact` proves that `0x77800000` is exactly `2^112`.
The native target is the independent scalar `FP.mul profile x constant`.
The selected model in the optional bridge uses nearest-even rounding with no
input/output FTZ; the public `kernel_fp_profile` covers every explicit `FP.Config`.
There is no arbitrary multiply function and no local numerical equality premise.
Overflow, subnormal handling, zero signs, infinities and NaNs are part of the
concrete scalar model; none is deferred to a numerical experiment.

Pinned offline compilation for SM80/SM90 and blocks 1/1024 emits predicated FP32
loads/stores, widened i64 indexing and `mul.f32 ... 0f77800000` without `.ftz`.
PTX multiplication without an explicit rounding modifier uses `.rn`. This is
selected-profile compiler evidence, not an all-block compilation claim or a
proof of GPU/compiler implementation conformance. The model's canonical NaN
behavior is explicit; no native NaN-payload conformance is asserted. Ordinary
successful compilation/resource/runtime conditions remain outside the theorem.

The Real theorem interprets memory through ideal arithmetic. It does not assert
that floating results equal mathematical Real values. The memory boundary is
aligned four-byte words with byte observations derived from those words; it is
not a new allocator or arbitrary byte-heap refinement theorem.

## Evidence and historical results

The workspace delivery freezes source pins, complete normalized source/DSL AST
equality, all-block actual-source CPU controls, classified original omission and
int32-wrap counterexamples, offline compiler artifacts, Lean receipts and the
transitive dependency closure. CPU execution is not GPU execution.

The previous host endpoints are preserved in `Misc/HostCorrect.lean` and
`Misc/HostFPEquiv.lean`, with their original explanations in `README.host.md`.
Their historical `LocalMultiplyExact` conditional result is not a premise or
numerical TODO of the new public kernel theorems.

## Implementation-pair review (2026-10-21)

The source-to-native results above are retained as ordinary theorems in
`Misc/NativeCorrect.lean`; their full names and statements remain unchanged.
`FPEquiv.scalar_left_equiv` now compares the complete selected masked source
against the existing scalar-left multiplication transformation. Both actual
sources are direct `triton` declarations in `Kernels.lean`; the Python file also
exposes `mul_kernel_scalar_left` with the same widened indices and masks.
The proof covers every complete program order, all supported block sizes,
empty inputs, tails and equal input/output regions, with successful execution,
source safety, typed output equality and both exterior frames.

The shared `FP.ScalarOpsLaws.mul_comm` theorem supplies exact multiplication
commutation for every FP32 word and every explicit `FP.Config`. NaNs, infinities,
subnormals and signed zeros require no finiteness restriction. The main result
is `original profile ... ≡[[]] scalarLeft profile ...`; it uses no experimental
rule and does not depend on the historical host `LocalMultiplyExact` premise.
This is a theorem in the declared scalar software model, with the same explicit
compiler/hardware boundary described above.

The independent `Correct.lean` mathematical target and its complete existing
input domain, source precision, launch coverage and frame results are retained.
No missing program connection is treated as a numerical assumption.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.


## Native body correspondence

`Kernels.lean` pastes the complete selected original and scalar-left Python
functions from `mul_exponent_compensator_repaired.py`. Both `.to(tl.int64)`
casts, the original `exponent_compensator: tl.constexpr = 2.0 ** (127 - 15)`,
all masks, loads and stores remain visible. Only BLOCK_SIZE is a call
constexpr; the declared runtime int64 length is bound outside the body.
`jit-correspondence.json` checks both complete bodies and external signatures.

The frontend folds only this generic exact-static-power subset to a weak
Python value, retaining its first interpretation in the original FP32 tensor
operation. Other floating powers and runtime exponents are rejected. Actual
signed mathematical index registers now match the displayed casts. The full
execution/memory and safety proofs transport those nonnegative addresses to
the old natural-index formulas. `Proofs/SourcePreservation.lean` constructs
both complete old/new executions and proves equality of all memory cells for
arbitrary floating callbacks and all old sizes. It does not claim equality of
their differently typed registers or ASTs.

The optional native decoder follows those actual signed casts into the same
existing finite index graph, using shared structural decoding in
`triton_mul2/Proofs/SignedSource.lean`. All original width, sign-extension,
allocation and grid-bound proofs are retained; no new per-task bit arithmetic
or numerical rule is introduced. Independent Real formula, full scheduled
outputs/frame and the strict software multiplication certificate are unchanged.
