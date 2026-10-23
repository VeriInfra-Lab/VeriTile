# matrix_reduction

The original wrapper generates input/output buffers in its explicit `float16`
or `float32` profile and launches one unmasked block-pointer kernel. FP16 loads
promote to FP32 for maximum reduction and convert back on store; FP32 retains
its source load/reduction/store precision. Both formats remain. The original
Python and its admission/failure behavior are unchanged.

`Kernels.lean` owns the readable typed Triton DSL used by both public proof
entries. `IO.lean` declares `matrixReductionIO`: the complete row-major
matrix input, one output per row, and no scratch. The `Correct.lean` headline uses the shared
`MaskedTileKernelIO₁.Implements` judgment and an independent finite maximum.
The maximum occurs in the row and bounds every row element. The judgment
connects actual source access safety, successful execution, all row outputs,
and the exterior frame through the existing flat-memory bridge. Positive N is
necessary; the mathematical theorem still permits M=0. Real erases rounding
from the same selected typed source, without changing native precision.

`FPEquiv.lean` now proves the actual original program equivalent to the
related `matrix_reduction_optimized.py` / `TypedSource.candidateKernel` source.
The candidate replaces the zero-offset full-tile block pointer with explicit
row/column affine addresses. This explores a simpler address lowering for the
same unmasked load; no measured performance improvement is claimed. Both
complete single-block programs retain FP16 promotion, FP32 maximum reduction,
the same reduction tree and native NaN/tie policy, final store conversion, every
row output, source safety and exterior frame. `matrix_reduction_pair` uses
`≡[[]]` for every primitive backend and all input words, with N>0 and M=0 still
admitted by the generalized DSL theorem. It introduces no numerical assumption.

`Misc/NativeCorrect.lean` retains the optional one-source native graph result.
The explicit bit-model `KernelFP.all_words` application is in
`Misc/KernelAllWords.lean`; `Proofs/GraphBase.lean` supplies only the task's
channel and scatter helpers needed by the main pair. `Kernels.lean` writes both
storage profiles directly as ordinary `triton` blocks.

The existing bit-level proofs are retained as supplementary results. They
prove exact half widening, inverse narrowing on the widened image, and maximum
selection in an explicit binary16/binary32 reference backend. The additional
non-NaN theorem gives the mathematical maximum interpretation of that bit
result, including infinities and subnormals. It is not an extra Python input
restriction, an experiment request, or a claim that the reference backend's
software NaN ordering is Triton's native NaN policy.

`Contract.lean` records the ordinary successfully generated Tensor storage:
shared metadata, selected dtype, shape, device, contiguity, absent lazy-negative
flag, actual data pointer, alignment and backing span. Both mathematical and
FP input bindings read the same physical addresses. `Proofs/KernelAddress.lean`
reuses the source's existing finite-address lemmas to show the block pointer
addresses coincide with those observations. i32 coordinates and strides are
extended before i64 multiplication; the ignored `stride_y` stays ignored.
No caller-provided Tensor wrapper or new guard/copy is introduced.

Successful native compilation/launch remains an API boundary. The unmasked
frontend requires supported block dimensions and at most 1048576 tile elements;
positive axes are needed for an actual launch. The generalized DSL theorem does
not claim unsupported shapes compile. In the inspected frontend, M=0 fails at
arange and N=0 fails in tensor-pointer rewriting. Those original failure paths
are not counted as successful kernel invocations. The source's generated output
allocation is contiguous and separate from the input; the existing flat IO
bridge accounts for physical placement.

Modeling evidence and citations remain in
[`fp_source_evidence.json`](fp_source_evidence.json) and
[`documents/GpuMemoryModel.md`](../../../documents/GpuMemoryModel.md).
Basic cast/max primitive semantics and random-generator API behavior are
modeling boundaries, not per-task numerical experiment TODOs. No native GPU
conformance or Real-versus-FP error bound is asserted.

Optional previous wrapper results are preserved in `Proofs/WrapperCorrect.lean`
and `Proofs/WrapperFP.lean`, reachable through the old compatibility entrypoints.
They cover the historical randn/empty/reference/assert/None-return composition;
the public kernel acceptance no longer imports or requires them. The reusable
independent maximum and finite-pointer lemmas were separated from host proofs.

Validation: `python bench/tritonbench_g/matrix_reduction/check_contract.py`
replays both actual source dtype profiles on real CPU tensors, checks source
binding, maximum results, exact launch arguments and unknown-dtype rejection.
CUDA allocation/launch are mocked. The targeted Lean build and standard-base
axiom audits are recorded in `/tmp/mean-matrix-reduction-kernel-delivery.json`.
No GPU execution or independent native replay is claimed.

The Real interface is `matrix_reduction_kernel_correct` in `Correct.lean`;
`FPCorrect.matrix_reduction_kernel_fp` in `Misc/NativeCorrect.lean` is an auxiliary
source/native result. `FPEquiv.matrix_reduction_pair` is the complete actual-program pair. These public
statements expose source precision, geometry, declared IO, and independent
mathematical/native targets. Each proof is one composition call; IO assembly
is in `Proofs/IOReal.lean` and `Proofs/IONative.lean`. State, execution and
cell-level details remain in `Proofs/KernelReal.lean` and
`Proofs/KernelNativeResults.lean`; the complete
arbitrary-state results are preserved there. The supplementary raw binary
reference result is `KernelNativeResults.matrix_reduction_bits`.
`Misc/TensorBridge.lean` retains `matrix_reduction_correct`, `matrix_reduction_fp`
and `matrix_reduction_maximum` under the `TensorBridge` namespace, with the
same actual input-storage bindings and optional non-NaN interpretation.
