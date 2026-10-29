# GEGLU tanh, selected FP32 kernel repair

The original fixed 128-column launch leaves columns 128 onward untouched when
`n_cols > 128`; the bundled width-256 case reproduces this source defect.
The separate existing Python repair uses a flat 128-lane grid and a global
count mask with signed-int64 index arithmetic. The original benchmark remains
unchanged. Successful repair proofs do not establish correctness of the buggy
original for wider rows.

## Target and source precision

Let `alpha = 0.7978845608028654`, `beta = 0.044715` and
`G(a) = a/2 * (1 + tanh(alpha * (a + beta*a^3)))`.
The independent Real formulas are `C[i] = G(A[i])*B[i]`,
`DA[i] = DC[i]*B[i]*G'(A[i])`, and `DB[i] = DC[i]*G(A[i])`.
The existing analytic derivative proof establishes that the written backward
formula is this gradient. The exact source decimal alpha is retained; Real
semantics do not replace it by the ideal `sqrt(2/pi)`.

Selected input/output storage is FP32, matching the benchmark profile and
existing checked repair. The source's FP32 casts, scalar order, decimal
constants and libdevice tanh are retained. Python's constexpr `3*0.044715`
folds to `0.134145` before FP32 conversion. The source frontend models each
constant operation with exact rational binary64 nearest-even rounding, then
checks the folded literal preserves both binary64 and FP32 conversion while
retaining the ideal Real projection. It does not reassociate runtime products.
The bounded constant subset rejects cancellation/zero, overflow, FP64-only
promotion and incompatible rounding; no arbitrary FP arithmetic identity is
assumed. No dtype family is added. The implementation follows Triton's
[`constexpr` arithmetic](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/language/core.py)
and [binary-expression dispatch](https://github.com/triton-lang/triton/blob/v3.7.1/python/triton/compiler/code_generator.py).

`Kernels.lean` contains the complete raw Python function bodies for forward, backward and
the existing integer-index commutation: `program_id * block` becomes
`block * program_id`. The scalar suffix is preserved verbatim. `candidate_kernels.py` spells the existing
index transformation in Python; the original repaired Python and wrappers are
unchanged. `jit-correspondence.json` checks all four complete bodies, their call
types and the actual `from triton.language.extra.libdevice import tanh` binding.
Pointer dtypes, constexpr arguments and import bindings are outside `triton { ... }`.
Both explicit int64 casts, the explicit FP32 input cast, comments and the
written `3 * 0.044715` product remain in each quotation. This is the
existing address-product transformation applied to the repaired flat launch;
it is not a new floating rewrite or a performance claim.

## Input contract and Tensor memory

The checked Python contract requires strided contiguous FP32 CUDA tensors,
no unresolved negative view bit, one device, equal element counts, a positive
last dimension, bounded launch/index arithmetic, aligned nonempty pointers,
and pairwise disjoint active input byte spans. Forward input shapes match;
backward A/B may be flat views matching the upstream element count. Empty
batches execute an empty kernel grid. The wrapper performs no materialization,
layout normalization, transfer or copyback.

The reused Tensor binding carries shared Torch metadata and maps logical
indices to the queried pointer plus `4*i`, including storage offsets. Physical
keys are device/address pairs, so same-storage slices and distinct overlapping
backing objects are allowed whenever their actual active spans are disjoint.
Forward's supplied fresh output has the ordinary `empty_like` metadata/bounds
contract. Backward mutates the supplied A/B windows in source store order.

`InputGuards.inputCheck_iff` proves exact Python input admission equivalence.
`DirectBinding.key_binding`, `indices`, and `pointer` connect that contract to
actual Tensor locations, signed native masks and aligned active addresses.
Memory API laws describe reads, writes and exterior-byte observations; they
supply no successful kernel execution or output fact. The shared load/store
model rationale is in
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md).

## Kernel-only proof entries

- `Correct.lean`: `geglu_kernel_correctness` uses the shared `ProgramIO.Implements`
  judgment for either source and any permutation of the complete grid. Its
  finite FP32 input/output channels expose forward C and backward in-place A/B;
  the independent Real formula gives every enabled output. Actual source access
  safety, successful execution and the full regional frame are proved together.
- `FPEquiv.lean`: `geglu_index_commuted_equiv` proves both actual shared sources
  successfully execute and agree on every typed memory cell. The integer
  prefix changes; the complete scalar suffix is identical.
- [Misc/NativeObservation.lean](Misc/NativeObservation.lean): the auxiliary theorem
  `geglu_kernel_fp32` uses the same shared IO judgment to connect
  either complete source grid to its independent native word formula, preserving
  the exact FMA/libdevice graph and all enabled FP32 outputs.
  `TensorBridge.geglu_kernel_output_bytes` exposes all four output bytes via the ordinary
  little-endian storage codec.
- `TensorBridge.python_checks_iff_preconditions` and `TensorBridge.checked_kernel_addresses` expose the
  checked-domain and actual native pointer/index connections.

`IO.lean` declares the source, finite channels, masks, in-place outputs, launch
order, Real IO and shared channel adapters. The independent Real formulas remain in
`Proofs/Real.lean` and `Proofs/Math.lean`; `Proofs/PublicIO.lean` supplies their
mathematical channel wrapper. `Misc/NativeIO.lean` holds the optional native
adapters and their complete IO theorem, using `Proofs/Native.lean` for the
independent native function.
`Proofs/AccessSafety.lean` checks the actual evaluated loads and stores for all
four source/direction selections. `Proofs/PublicIO.lean` composes those facts
with the existing `Tile`, `Grid`, `Math` and rewrite results.
`Proofs/Kernel` contains the source-execution relation and grid output/frame
results. Existing launch metadata is reused as `Prepared`; its ordinary buffer
facts are not a wrapper execution premise. Public results contain no runtime
failure branches, allocator algorithm or returned-Tensor claim. Former host
entries remain optional in `Proofs/HostReal.lean` and `Proofs/HostFP32.lean`.

## Numerical status and validation

There are no numerical transformation assumptions or deferred experiments for
the index commutation. Native ternary FMA, FP32 primitive operations and the
unchanged libdevice tanh branch graph retain their exact instruction ports,
including FTZ sites and NaN-sensitive comparisons. They are ordinary model
boundaries. The historical `PrimitiveObligations` interface remains an optional
model-conformance result; it is absent from current public kernel premises.

CPU source checks pin the original scalar AST suffix, reproduce the width-256
source defect, exercise ten forward/in-place-backward paths (tails, wide rows,
3-D shapes, offsets and empty batches), fourteen rejected inputs, and fourteen
actual index-commuted AST cases with exact backing bytes and exceptional FP32
words. These use CPU primitive adapters and mocked CUDA; they are not native
GPU numerical experiments. GPU execution and independent native replay are
not claimed.

The current targeted Lean build covers the IO definitions, actual-source safety,
public composition, physical input binding, both public entries, retained raw
results and `TensorBridge`. Earlier builds also covered `Proofs.HostReal` and
`Proofs.HostFP32`. The independent Real
formula and compiled native reference do not depend on the kernels or source
scalar evaluator; the public proofs use only the standard axiom base; and the
index equivalence reports no floating-point assumptions.

`Correct.lean` and `FPEquiv.lean` retain the complete forward/backward kernel certificates and index-commuting equivalence. `Misc/TensorBridge.lean` contains the optional Python checks, checked address facts, and byte-observation corollary as ordinary theorems in `GegluTanhTriton.TensorBridge`.

## Raw kernel interfaces

The previous arbitrary-carrier, arbitrary-`Ops` physical certificates remain
unchanged as ordinary theorems in `Proofs/RawCorrect.lean` and
`Misc/RawNativeApplication.lean`, including every byte of the exterior frame.
The source-pair connection stays in `Proofs/RawNative.lean`; it does not import
the optional raw native application.
`Misc/PhysicalIO.lean` binds the exact finite channels to these same physical
heaps and their capacities; it adds no canonical-heap or stronger alias
restriction. `Proofs/RawKernel.lean` and the original source certificates retain
the physical execution/byte transport foundation. `TensorBridge.geglu_kernel_correctness` and
`TensorBridge.geglu_kernel_fp32` retain the original admitted Tensor applications.
Their existing `Kernel` adapter and the raw interface reuse the same source
certificates and physical-grid results, with `DirectHost.layout` and
`key_binding` providing the correspondence. Precision and native ports are
unchanged.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeObservation.lean](Misc/NativeObservation.lean), [RawNativeApplication.lean](Misc/RawNativeApplication.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

## Source readability cleanup (2026-10-22)

The selected forward/backward original and index-commuted kernels are now four
complete ordinary `triton` quotations in `Kernels.lean`. The previous whole-source
macro is removed. Complete function equality with the preserved pre-cleanup source,
including all metadata and typed statements, is checked by `rfl` in the scoped
source-only evidence. All masks, constants, libdevice tanh, casts, floating order,
in-place stores, named interfaces and input domains are unchanged.

The scoped source-only delivery records targeted dependency builds and a fresh
joint check of the existing public and optional entries. This organizational
change does not reopen proof completion, change contracts or add a numerical,
host, compiler or hardware acceptance requirement.
