# Contiguous FP32 softmax repair

The canonical source is `triton_softmax_repaired.py`. Its independent mathematical
target is `exp(x[r,c]) / sum_j exp(x[r,j])`. The original benchmark and canonical
Python source are unchanged by the kernel-proof refactor.

The checked domain remains rank-two, contiguous, strided FP32 CUDA Tensors without
an unresolved negative-view bit. Nonempty inputs require a four-byte-aligned
actual data pointer, fewer than `2**31` rows, and a next-power-of-two column block
at most `2**20`. There is no total-element cap beyond valid Tensor storage.
Nonzero storage offsets, unusual irrelevant singleton strides, empty dimensions,
NaNs, infinities and subnormals remain admitted. Empty axes cause no pointer query
or kernel launch. `TensorBridge.python_checks_iff_preconditions` in optional
`Misc/TensorBridge.lean` proves the two-way
correspondence between the Python checks and the input contract.

The historical repair removes the original 1024-column cap, passes both column
strides, and widens program/lane indices and runtime strides to int64 before
address arithmetic. It preserves the separate `row_max` temporary, all floating
assignments, and the omitted `num_warps` argument, which selects four warps at
every width. The kernel computes a global maximum over the complete padded row
before exponentiation. In particular, a row containing only negative infinities
before a finite last coordinate produces mass at that last coordinate; it does
not use the online recurrence that can poison an earlier empty tile.

`Kernels.lean` contains the complete source DSL with explicit FP32 input/output
regions. Complete Tensor application entry points in `Misc/TensorBridge.lean` are:

- `TensorBridge.Real.softmax_kernel_correct`: actual successful source execution, every
  logical output column of any valid row at the output Tensor's physical index,
  the independent unshifted Real softmax formula, the complete exterior-cell
  frame, and preservation of the input allocation.
- `TensorBridge.Native.softmax_kernel_fp32`: actual successful typed source and candidate
  executions with the same entire state and raw typed-cell exterior frame,
  their common physical result, every output word equal to the independent ordered native graph on the original
  logical input words, and the complete cell frame and input preservation.

Both theorems explicitly skip zero-column launches. Their invocation metadata
supplies a same-shape output Tensor in separate storage with the ordinary valid
contiguous FP32 layout. They prove each valid kernel row and all its columns;
allocation and host-wrapper execution remain in the historical proof entry
points below. Neither public entry imports the host wrapper or allocator.

`KernelBinding.source_load` composes three independently defined observations:
the actual backing-storage cell, the widened native byte-pointer reader, and the
original Tensor logical coordinate. Both Tensor application proofs consume this composition.
Storage-derived region identity and `Torch.FP32.TensorLaws` retain aliases,
physical indices, storage offsets and contiguity. `pointer_bounds` establishes
finite native pointer arithmetic, allocation bounds and four-byte alignment.

The separate candidate `triton_softmax_index_commuted.py` changes exactly
`row_idx * input_row_stride` to `input_row_stride * row_idx`. Its DSL translation
is `indexCommutedKernel`. `KernelRewrite.execution` derives equality of complete
`FP.Structural.exec` results using `equiv_decompose` and `fp_prove` with an empty
numerical-assumption table. `KernelNative.complete` combines this result with
source termination, exposing the same successful typed state and its actual
owner-thread projection. It does not replace source execution with a reference
graph callback.

The native model uses the task's fixed-four-warp graph and production layout V1
for runtime-int64 strides. It retains register left folds, simultaneous
self-first warp butterflies, shared-memory warp aggregation, subtraction,
multiplication by FP32 `0x3fb8aa3b`, `ex2.approx`, and `div.full`. Six scalar raw32
instruction ports are explicit: `maxNum`, `subRN`, `mulRN`, `exp2Approx`, `addRN`,
and `divFull`. Source/candidate equality holds for every common choice of these
ports, including all raw FP32 inputs and padding. It needs no associativity,
finite-input, scalar-accuracy or FP32-to-Real rewrite assumption. There is no
deferred numerical experiment obligation for this kernel-equivalence result.

The external model boundary is Tensor API/storage laws, source-to-DSL
correspondence, FP32 word encoding, selection of this compiler graph and common
primitive instructions, and a successful race-free GPU runtime. The Real theorem
uses the erased mathematical semantics. No equality between FP32 and Real softmax,
verified compiler theorem, hardware instruction conformance, or GPU experiment is
claimed.

Historical wrapper, allocation, grid-composition and exact-returned-object proofs
are preserved under `Proofs/HostCorrect.lean` and `Proofs/HostFPEquiv.lean`, in the
`HostCorrect` and `HostFPEquiv` namespaces. Their scalar-instruction agreement
transport theorem remains separate from the unconditional candidate result.

Validation uses direct Lean compilation into the workspace cache, combined
public imports, non-circular specification checks and standard-base axiom audits.
`check_contract.py` exercises source correspondence, guard order, 16 CPU Tensor
admissions, 14 rejections, offsets/singletons, empty out-of-storage views, pointer
alignment, fixed-four-warp launch metadata, large shapes, caller preservation,
and original transpose/stride/overflow and 1025-column tail defects.
`check_index_candidate.py` checks the exact single-product AST rewrite and 36
complete-storage bit comparisons over finite and exceptional values at widths
1 through 32769, including block boundaries, nonzero offsets, input preservation
and output canaries. Three independent negative-infinity-prefix controls check
that the sole finite last coordinate receives exact mass one. CUDA launch and
allocation are mocked for these CPU source checks.

## Public IO and Tensor applications

`IO.lean` declares the selected FP32 quote, arbitrary storage offsets,
row/column strides and active load/store windows. `Correct.kernel_real` uses the
shared `MaskedTileKernelIO₁.Implements` judgment: every active logical output is
the independent unshifted softmax, with successful flat execution and exterior
frame. `Proofs/RowIO.lean` connects this IO to the actual source trace safety and
reuses the established row arithmetic proof. `Proofs/RealIOImplements.lean`
assembles this result; `Correct.lean` retains the independent formula and a short
composition proof. The public block-cover and output
address injectivity conditions preserve the original domain.

`IO.nativeIO` declares the corresponding typed regional IO for either the
source or the retained integer-address candidate. Its fixed four-warp primitive
backend, launch IDs, active windows, FP32 input encoding and output-thread
observation are explicit. `FPEquiv.kernel_native` now compares the two actual
complete row invocations as `original ≡[[]] indexCommuted` through the shared
`ProgramIO.Implementation` view. This reuses both established native IO proofs
and retains actual source access safety, successful executions, every observed
native output word and both complete raw-cell frames. The existing domain
(shape agreement, output injectivity, positive covered block and valid row) is
unchanged. The former per-candidate `ProgramIO.Implements` theorem and its
independent ordered native graph are retained in `Misc/NativeCorrect.lean`. `Proofs/StructuralAccess.lean` proves actual load/store trace safety
for both sources; `Proofs/NativeIO.lean` connects the stored tagged cells to the
native formula. `Proofs/NativeIOImplements.lean` holds the IO assembly, leaving
`Misc/NativeCorrect.lean` with the independent plain-array formula and native
correctness assembly. `Proofs/Pair.lean` composes the two native implementations,
leaving `FPEquiv.lean` with the explicit comparison and short proof. The Real
and auxiliary native declarations retain their statement surface checks. The regional native IO does not claim an automatic flat-pointer
transport theorem.

The former raw reader statements remain as ordinary supporting results in
`Misc/KernelResult.lean` and `Misc/KernelNativeResult.lean`. The latter retains
the exact common source/candidate state and its physical owner projection.
`KernelRewrite.equivalent` remains the standard empty-assumption program
transformation proof. None of these results replaces the independent native
formula with mathematical Real arithmetic.

`TensorBridge.Real.softmax_kernel_correct` and
`TensorBridge.Native.softmax_kernel_fp32` retain the full checked Tensor results,
including empty-column no-launch, physical indices, source/candidate common
states, projected native words and input preservation. Their proofs share the
same established source execution and output lemmas; the reader entries import
no TensorBridge. Python, the source quote and the input contract are unchanged.


## Native body correspondence

`Kernels.lean` pastes both complete selected Python bodies, including the
program-id/column int64 casts and the named row_max assignment. The original
`triton_softmax_repaired.py` and existing `triton_softmax_index_commuted.py`
remain unchanged; the candidate only commutes the input row-address product.
Runtime int64 stride/count argument values are supplied as signed Int carriers
outside the body. Only BLOCK_SIZE is constexpr.

The original pointer arguments already point at their Tensor view bases.
`pointer_offsets` supplies those allocation-relative values at the call site,
without inserting the older presentation's extra offsets into the pasted
function body. The bindings use the same element-offset coordinates and full
arbitrary-stride/offset scope as the original logical and physical IO. The
source checker records every offset and rejects changed, missing, duplicate or
nonpointer bindings. The frontend rejects reassignment of offset-bound pointer
arguments in this supported subset instead of silently rebinding them.

All complete actual executions, masks, access bounds, mathematical row formula,
floating graph, source/candidate equivalence and full memory frames are
re-proved. Exact signed index commutation and address-projection congruence use
shared logical lemmas; they introduce no floating numerical law. Existing
native index/byte, Tensor, full-wrapper and model assumptions retain their
original scopes. The selected intrinsic, arithmetic order and four-warp default
are unchanged; no new GPU or numerical experiment is added.
