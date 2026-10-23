# Checked FP8 conversion kernel contract

The canonical scope is the existing checked repair. Its Python and input
checks are unchanged: packed int8 decode or FP16/FP32 encode, CUDA, strided
contiguous layout, no lazy negative flag, natural alignment, and E5/E4NV
format. Scalars, empty tensors, nonzero storage offsets, and arbitrary total
counts remain admitted. No finite-value or new count guard was added.

The original wrapper remains a source defect. For `backing[2:5]` from eight
FP32 elements it passes `N=8` to a three-byte output. A stride-two input is
also admitted, although the kernel loads consecutive elements. Original
`Proofs/SourceDefect.lean` proves these metadata/address contradictions and
the signed-i32 offset overflow; `check_original_defects.py` executes the
actual original wrapper AST for the slice and stride cases.

There is one authoritative readable typed DSL source in
`bench/tritonbench_g/f8_conversion_utils/Kernels.lean`. The repair imports it.
Both original decoder stores, FP8 E5/E4NV, FP16/FP32 input, destination casts,
masks, and unspecified masked load values are retained. All six selected conversion profiles and their existing index candidates are
written as complete ordinary quotations; their declared storage types are unchanged.

`Contract.lean` publishes the exact executable admission, independent
`conversionReal` and `conversionBits` formulas, and `TensorStorage`. That
storage binding records the allocation, storage offset, logical-to-physical
indices, and the ordinary contiguous Tensor API law. The input data pointer
includes the storage offset exactly once. These are storage facts, with no
kernel result or execution premise.

`Correct.lean` displays `conversionIO : MaskedKernelIO₁` and its Real
identity specification. The shared IO assembler proves active source access
safety, successful flattened execution, logical outputs and the complete frame.
`Misc/NativeCorrect.lean` retains the typed `ProgramIO`: source-selected storage channels,
masked windows, actual kernel invocation, byte encoding and observation.
Its independent `bitFormula` admits all exceptional inputs and both decoder
stores. `Proofs/KernelNative.complete_native_kernel_refinement` connects every
actual checked chunk program to the independent native byte program. Existing
chunk proofs establish i32 arithmetic for every launched lane and 64-bit
physical addresses for active loads/stores; the total Tensor count is not
restricted. Public logical Tensor results connect initial logical values,
actual final memory, output coordinates, and the exterior byte frame.

The bit machine defines its primitive conversion by exact rational nearest-even
selection with finite saturation for encode and exact finite decode. It retains
both zero signs, subnormals, infinities, and a class-preserving NaN policy;
`canonicalPolicy` is a concrete inhabited policy. The theorem has no
`LocalCast`, finite-domain, rounding-conformance, or successful-execution
hypothesis. Primitive IEEE/FTZ/NaN/compiler behavior is model scope, not a
per-task experiment. The deferred optimization-law list is empty. The native
word graph is independently written; no identity candidate was introduced.

The original natural-index theorem alone is not an unrestricted native launch
claim. Native coverage here comes from the unchanged repair's actual chunk
schedule. Hardware implementation conformance to the selected primitive model
is not proved, and no GPU was run. Existing CPU interpreter and SM90 offline
lowering receipts remain historical diagnostics, not numerical-law evidence.

`Misc/TensorBridge.lean` is the optional interface for checked wrapper results, logical Tensor applications, input-guard correspondence, and checked pointer/alignment/outcome helpers. They are ordinary theorems, with identical helpers shared once. `Correct.lean` exposes the independent Real IO specification. `FPEquiv.lean` compares the two complete typed source programs by `original ≡[[]] indexCommuted`, reusing the existing six-profile integer-index proof. The bit-formula implementation is an ordinary theorem in `Misc/NativeCorrect.lean`, preserving its qualified name. The former raw execution theorem is retained in `Proofs/KernelReal.lean`; the bit execution, native refinement and complete native byte frame are retained as ordinary theorems in `Proofs/KernelNative.lean`. `Proofs/SourceSafety.lean` checks actual Structural loads/stores in all six profiles; window declarations alone are not used as a safety proof. The primitive cast/NaN/rounding policy and absence of deferred optimization laws are unchanged.

The 2026-10-21 pair review retains all six source profiles and reuses the
original task's index-commuted `toHalf`/`toFP8` candidate. Only natural index
multiplication operands change. This exact source proof adds no floating rule, cast
assumption or numerical experiment. The retained native and Tensor byte results
continue to use their explicit conversion policy and actual checked chunks.

## Optional proof layout (2026-10-21)

Optional native, Tensor, host and audit applications are in `Misc/`; import them
explicitly. Declaration namespaces, theorem statements, proof bodies and the
program-pair review status are preserved. The required entries do not re-export
optional native conclusions. This file organization adds no mathematical or
numerical completion claim.

The 2026-10-22 source-only cleanup replaces the canonical whole-source macros
with complete ordinary `triton` quotations. Compatibility spellings now delegate
to those named sources. `Proofs/SourcePreservation.lean` proves complete function
equality to the frozen former source for all 12 typed original/candidate
configurations and all four FP64 erasure controls, including input/output
signatures, both decoder stores, omitted load padding and every parameter.
The frozen generators are proof-only and are not imported by the source or
public proof entries. This organization change preserves the existing proof
status, checked input scope, native policy and empty numerical-obligation list.
