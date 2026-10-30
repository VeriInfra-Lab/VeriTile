# kldiv_triton: checked contiguous FP32 contract

The current entries are `kldiv_forward_triton` and `kldiv_backward_triton` in
`kldiv_triton_repaired.py`. The original benchmark is unchanged.
`kldiv_triton_conditional.py`, `Proofs/ConditionalCorrect.lean`, and
`Proofs/ConditionalFPEquiv.lean` preserve the historical generalized repair and
its valid results. Its exact description is retained in `README.conditional.md`.
The new public namespace is `VeriTile.Bench.Repairs.KldivTriton.Contiguous`.

## Supported API and source changes

Prediction, target, and the supplied backward destination are matching rank-two
strided FP32 CUDA tensors. Every tensor argument must be contiguous, have no
unresolved negative view bit, and be on the same device. Nonempty tensors must
have a four-byte-aligned actual data pointer. Nonzero storage offsets and
contiguous degenerate strides are accepted; neither input is copied.
`log_target` must be a Python `bool`; forward reductions are exactly `none`,
`sum`, `mean`, and `batchmean`. Backward cotangents are FP32 tensors of scalar
shape or exactly the matrix shape. All FP32 words are admitted: there is no
positivity, finiteness, or other tensor-value guard.

For a nonempty backward destination, its complete backing storage must be
disjoint from both the target and cotangent backing storage. This deliberately
conservative storage check also rejects disjoint views into one shared
allocation. It permits direct writes while preserving both read roles. Empty
destinations skip the alias check because they perform no writes. Forward read
inputs may alias arbitrarily. All rank-two empty shapes are admitted.

The two JIT kernels are AST-identical to the preserved historical candidate.
They retain the necessary source repairs:

- Raw target-zero terms are selected to literal zero; padded lanes are selected
  to zero before reduction.
- Reduced output shape is `(rows, ceil(cols/block))`, backing every actual chunk
  store. The chunk loop remains part of the source.
- `mean` is full sum divided by `rows*cols`; `batchmean` divides by `rows`.
- Signed64 source indices/strides and positive block selection also cover tails,
  multiple chunks, and empty width.

The checked forward host passes the two original tensors directly and returns
the actual zero allocation for `none`, or the actual Torch scalar sum/division
result. Backward passes the supplied destination directly to the kernel, writes
`-target` or `-exp(target)`, then returns that same object for scalar-one or the
actual product with the cotangent. No resolve, clone, packing, work buffer,
copyback, or added reduction normalization occurs. Both use the actual tensor
owner's CUDA device context.

## Admission and physical interpretation

`Contract.lean` records exactly the observed Python metadata and storage
intervals. `python_forward_checks_iff` and `python_backward_checks_iff` prove
both directions between ordered rejection checks and the public contracts,
including non-Tensor/flag/reduction classification. `MatrixBinding` and
`ViewBinding` are ordinary Tensor metadata, storage, active-pointer, backing,
and contiguous-view laws. They contain no kernel-output equality.

A contiguous singleton column may retain a nonunit unused column stride.
`MatrixBinding.store_address` proves the source's actual unit-column address
only on active coordinates. `source_pointer` derives the actual checked pointer,
four-byte alignment, bounded native64 arithmetic, and all four backed bytes.
`checked_indices` derives signed64 masks and loop increments from actual backing.
There is no added `numel` cap. Ordinary signed64 stride metadata and successful
native launch-ID bounds remain explicit runtime contracts. The shared launcher
compatibility item in `bench/TODO.md` remains deferred; no large-grid GPU
compatibility claim is added here.

`AllocationContract` records the ordinary four-byte alignment of fresh Torch
allocations. `Launch.Contract` records successful setup's ordinary metadata/PID
bounds; neither assumes a kernel result. The FP public entries state both
contracts explicitly. `forward_complete` and `backward_complete` prove that
complete source row executions at their evolving physical memories equal the
source-derived store graphs used by the host proofs. The direct backward proof
uses checked storage separation, including the empty case, and does not rely on
historical work-buffer or snapshot assumptions.

## Public results

`Correct.forward` and `Correct.backward` prove the checked APIs in the independent
Real interpretation. Raw terms are zero at target zero and otherwise
`t*(Real.log(t)-y)`; log-target terms are `Real.exp(t)*(t-y)`. The full logical sum
and requested normalization are proved independently of chunking. Lean Real's
total log and division conventions do not assert accuracy or exceptional-value
agreement with native execution. Backward has no reduction argument, so the
caller supplies any normalization through the cotangent.

`FPEquiv.forward` and `FPEquiv.backward` prove the complete FP32 raw-word graph
and four output bytes, the actual supplied/returned object, the caller byte
frame, and successful context restoration. The actual local reduction tree and
Torch-selected `ReductionPlan` remain structural parts of the execution.
`runForward_kept` and `runBackward_kept` quantify over arbitrary initially live
allocations and preserve their entire bytes, or the complete complement of the
supplied destination. Failure prefixes retain the same exterior frame, including
errors after the derivative has already been written.

The native graph preserves the frozen compiler profile's loss multiplication
operand order, installed `tl.log` intrinsic, exp scale then `ex2.approx`, raw
literal-only sign XOR, and log-target backward subtraction from positive zero.
No FP reassociation or commutativity assumption is used. Source/load/store and
ordinary runtime laws remain explicit; this is not a verified Python compiler
or a proof of the GPU compiler/runtime implementation.

## Numerical boundary and validation

`FPEquiv.forward_reference` and `FPEquiv.backward_reference` use only fixed-arity
scalar obligations in `Native.Obligations` and `NativeHost.Obligations`. The
registered domains are all FP32 operand words, and actual Python natural
normalization denominators for Torch scalar division. There is no hidden
finite/positive input restriction. Triton division exists in the generic scalar
interface but is unused by these kernels. No whole-output, whole-sum, mask,
address, execution-success, or frame property is deferred as numerics.
`numerical-obligations.json` records the local declarations and required native
evidence. GPU numerical conformance remains deferred; an IEEE NaN-payload or
correctly-rounded intrinsic claim is not made.

Targeted Lean compilation and standard-base axiom audits pass for the restored
historical modules and new checked modules. `check_contract.py` verifies source
pins and exact historical JIT AST equality, 33 rejection cases, 5 ordinary
failure prefixes, and 134 actual CPU source/reference/frame cases. These include
empty axes, width 32769, all modes/flags, scalar/full cotangents, offsets,
degenerate contiguous strides, overlapping forward reads, partial allocation
tails, exceptional values, direct arguments, and return identity. CPU adaptation
changes device metadata/context/allocation routing only; no GPU is queried.
Four separate original-source counterexamples reproduce padded NaNs, target-zero
NaNs, the wrong mean denominator, and the second chunk's out-of-view write using
explicit sentinel backing.
