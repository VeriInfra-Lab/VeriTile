# rmsnorm_triton kernel revision

The selected source is `rmsnorm_triton_repaired.py` from `task_progress.json`,
SHA-256 `5b89dde21e5b2752f1af7aaee97317c7a55824a08901a480fffdee875164cf0c`.
Its production JIT AST is identical to the original source. This revision changes
Lean contracts and proofs only.

`Kernels.lean` displays the complete typed FP32 JIT. It retains both 1024-lane
loops, the FP32 input conversions, `libdevice.pow(x, 2)`, zero input padding,
weight padding without an `other` value, the sum, division, epsilon addition,
rsqrt and both output multiplies. `indexCommutedKernel` changes only the order
of the first integer address multiplication.

`Contract.lean` preserves every Python guard: contiguous rank-three FP32 CUDA
input, a matching FP32 weight vector on the same device, no lazy flags and a
finite positive Python-float epsilon. A positive feature width is required.
Empty leading dimensions skip launch checks. Active launches check the padded
final increment, every actual stride and maximum formed offset against signed
i32 limits, all queried pointers for four-byte alignment, and all padded pointer
formations for u64 wrap. Singleton dimensions retain their actual strides,
including zero strides. Offset views and input/weight storage aliases are
admitted. Output freshness and active coordinate injectivity are ordinary
allocator facts, not numerical or execution assumptions. The kernel boundary
also records the ordinary `empty_like` output-device equality. Inputs and
weights explicitly belong to the live allocation set, so physical separation
cannot be weakened by an empty live-set parameter.

`IO.lean` exposes the complete grid as a shared `UKernelIO`: one width-N
input/output channel per row, a shared weight channel, all original strides,
and no scratch. `Correct.kernel_correct` uses
`RegionalScheduledImplements` over the exact arbitrary permutation of the
full 2D grid. Its independent `RmsIO.rms` formula contains no interpreter or
source recurrence. `IOSafety` proves actual source trace safety from those IO
windows, and `IORow`/`IOGrid` compose the established two-pass theorem into
successful whole-grid execution, every output, and the raw-cell frame. The
canonical safety helper now accepts active output injectivity; its previous
positive-stride interface is retained unchanged. Empty grids and singleton
zero-stride rows remain included.

`Proofs/KernelRealResults.lean` and `Proofs/KernelNativeResults.lean` preserve
the exact prior raw physical-grid statements. `Misc/TensorBridge.lean` retains its
original statements and uses the same existing execution foundations. The
public readers and every transitive helper are independent of public readers
from other tasks. Native source equivalence is the single FPEquiv headline;
no additional numerical law or GPU experiment is introduced.

`TensorBinding` contains actual `VeriTile.Torch.FP32.Tensor` objects and their
ordinary API laws. Logical-coordinate reads, allocation identities, physical
indices, byte bases and actual `dataPtr` observations are used in the public
conclusions. `StorageBinding` ties native capacities to those same actual
allocations; `ActualStorageSafe` proves bounds and all four bytes of every
active access. No storage alignment law is assumed. `Observes.pointer` applies
only to nonempty tensors, preserving the early-empty behavior of PyTorch.

`TensorBridge.kernel_correct` establishes the independent mathematical formula
`x[b,m,k] / sqrt(sum_j x[b,m,j]^2 / K + eps) * weight[k]`, all output values,
successful typed source and candidate grids, complete physical-memory equality
to an independent output update, the full
exterior frame, actual Tensor pointer equations and checked machine-address
correspondence. The typed interpreter proof reuses the full two-pass invariants;
its only strengthened premise is active-coordinate injectivity in place of a
positive output stride, so singleton zero strides remain supported.

`TensorBridge.kernel_native_equivalence` proves successful execution of the actual
typed source and candidate grids under the symbolic FP Structural interpreter.
`FPSource.complete` executes both loops, including all loads, casts, the native
integer power call, reduction, arithmetic and masked stores. `FPValues.graph_spec`
proves each output from an independent logical expression recurrence and proves
the full frame using active-coordinate injectivity. `FPPhysical` composes those
programs into physical-memory rows and arbitrary complete grid orders.
`FPObserved.tensor_result` then projects the successful symbolic heap to FP32
words and proves every actual Tensor output and full-memory equality to the
independent reference update. Source success is a proved conclusion, and no
handwritten native-machine success is substituted for it.

`FPGraph.Profile` makes reduction layout, initial add order, output multiply
operand order, the two FTZ multiplies for integer power, and width-one division
specialization explicit. Its observer examines expression structure; it never
assumes FP commutativity, a zero identity, a multiply-by-one identity, or a
Real-to-FP equality. The independent target `nativeTensorResult` uses only actual
logical input/weight words, a pure lane recurrence and that observer. NaNs,
infinities, subnormals and signed zero remain admitted. Scalar IEEE/FTZ primitive
semantics and the selected compiler profile are explicit parameters; the
source-to-symbolic-to-observed output and memory bridge is proved in Lean.

`FPEquiv.source_candidate_equivalence` also proves complete typed source/candidate
state and failure equivalence for arbitrary structural backends. It has zero
numerical assumptions: the candidate changes an integer address expression
only. No floating reassociation, approximate identity or FMA replacement occurs.

`Proofs/Facts.lean` proves denominator positivity, empty-grid identity, singleton
zero-stride injectivity and the counterexample: width 2147482625 passes K<2^31
but its final padded increment has signed value -2147483648.

Validation uses the existing read-only Lean cache with the workspace overlay:
`python /tmp/veritile_workspace_lean.py <task modules in dependency order>`.
No new Lake build, cache removal, native GPU execution or numerical experiment
is needed or claimed. The inherited host proofs are compatibility artifacts;
the public proof endpoints in this revision are kernel-only.

The main `Correct.kernel_correct` takes kernel geometry, separation, active
output uniqueness and a complete schedule, and proves the independent Real
formula, source success and exterior frame. The separate main
`FPEquiv.source_candidate_equivalence` compares the actual typed source and
integer-index candidate with an empty numerical rule set. The native and
reference-memory results remain auxiliary; `Misc/TensorBridge.lean` retains
the full prior Tensor checks, logical readback, actual addresses and storage
observations. The Real grid proof and Tensor bridge reuse
`Proofs/KernelGrid.lean`; the bridge explicitly projects the same source from
kernel address geometry. No Tensor/API metadata is a main argument, including
inside the geometry records.
