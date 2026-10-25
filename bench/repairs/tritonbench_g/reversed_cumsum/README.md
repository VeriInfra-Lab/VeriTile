# reversed_cumsum: kernel correctness and triangular comparison

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/OptionalHostCorrect.lean](Misc/OptionalHostCorrect.lean), [Misc/OptionalHostFPEquiv.lean](Misc/OptionalHostFPEquiv.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The independent mathematical result is
`Z[b,h,t,s] = sum(input[b,h,k,s], t <= k < T)`.
The required proof scope is one kernel program, following the approved
`add_value` standard. Wrapper assertions establish the supported input domain;
allocation, autotuning and returned-object composition remain optional.

## Supported inputs and precision

The wrapper admits rank-four strided, contiguous CUDA FP32 Tensors without an
unresolved negative view bit. Output dtype is None or FP32. Nonempty inputs
need a four-byte-aligned data pointer and both `B*H` and `ceil(S/32)` below
`2**31`. Empty axes return after allocation without querying an unused pointer
or launching. Offset views and unusual singleton strides are supported;
noncontiguous views are rejected. NaN/Inf, signed zeros and subnormals remain
admitted. There is no divisibility or finite-input guard.

[Contract.lean](Contract.lean) names each assertion.
`TensorBridge.python_checks_iff_preconditions` proves exact dynamic admission,
including non-Tensor rejection. `Proofs/KernelContract.lean` gives the ordinary
shape, original input-pointer and FP32 storage binding, and derives native PID
and alignment facts without assuming execution or output values.

All nine original tuning choices remain: `BT` in 16, 32 or 64, each with 2, 4
or 8 warps; `BS=32`. The source retains its FP32 triangular zero/one matrix,
FP32 load conversion, `dot(..., allow_tf32=False)`, incoming-carry addition,
store-before-reduction order, configured axis-zero reduction, descending carry
update and FP32 output conversion. No reduction-order rewrite is introduced.

## Shared source and kernel mathematics

[Kernels.lean](Kernels.lean) contains the authoritative readable typed DSL pair
and [Contract.lean](Contract.lean) contains the independent ordinary suffix formula `mathematical`. `Kernels.source`
is the production repaired kernel. `Kernels.compareRight` uses
`o_i[None, :] >= o_i[:, None]` where the production source uses
`o_i[:, None] <= o_i[None, :]`. This is the task's intended triangular-comparison
variant, with every floating instruction and all other source statements
unchanged. Its actual Python JIT body is
`chunk_global_reversed_cumsum_vector_kernel_compare_right` in
[reversed_cumsum_repaired.py](reversed_cumsum_repaired.py). The wrapper still
selects the production source.

[Correct.lean](Correct.lean) and [FPEquiv.lean](FPEquiv.lean) use exactly that
same production source; FP equivalence retains the typed body. Only the Real
interpretation erases dtypes. Native i64 casts are projected to mathematical
indices in the displayed DSL. `TensorBridge.native_programs` certifies the
fixed-width PIDs, reverse-loop control, final decrement, clamped inactive
coordinates, masks, offset products and byte-pointer windows from the checked
input and ordinary allocation bounds.

`Correctness.kernel_correct` uses the examples-style
`Spec.Real (io h c program ⊨ fun xs _ j => ...)` and directly displays the
ordinary suffix sum for one row and feature-tile program. [IO.lean](IO.lean) exposes
the actual shared source, explicit Real dtype erasure, input/output regions,
`T × 32` channel windows, feature-tail masks and the two selected launch axes.
The displayed formula is definitionally equal to the retained independent
`reference`; the proof reuses `IOComplete.implements`. The flattened logical
lane `j` has time `j.val / BS` and feature `j.val % BS`; `lane` selects the
same feature at each time in the suffix. There is no private scratch or hidden
execution premise. The shared
`AlgorithmProgramIO.Implements` judgment includes successful execution,
actual source access safety, all active outputs and the exact exterior frame.
[Proofs/SourceSafety.lean](Proofs/SourceSafety.lean) proves access safety for the
actual repaired masked-pointer body, using its reached reverse-loop states.

`TensorBridge.kernel_correct` preserves the previous physical result through
[Proofs/KernelCompatibility.lean](Proofs/KernelCompatibility.lean).
`Storage.Placement` supplies disjoint allocation windows, active bounds and
native address-space limits. This consequence retains every suffix output,
input-allocation stability and every exterior physical element cell. Both
results reuse the same `RealExecution.source_correct` foundation. Physical
cells are keyed by byte addresses; this is not a raw-byte encoding theorem.

## Actual FP equivalence

`Equivalence.reversed_cumsum_compare_right_equiv` proves equality of complete
typed execution states and failures for the two displayed sources, for all
initial states and arithmetic backends. The proof reverses only a natural
comparison; it changes no FP32 dot, reduction, rounding order, conversion,
mask or store. Its numerical assumption list and rewrite-domain guards are
empty. The assumption audit reports **none**.

`Equivalence.successful_pair`, in
[Proofs/FPCompatibility.lean](Proofs/FPCompatibility.lean), separately
constructs successful runs of both complete sources for any admitted program and total configured dot backend,
with the memory frame retained. It supplies no execution-success premise.
This kernel equivalence does not require primitive native-to-IEEE conformance,
and no numerical transformation experiment is deferred. Native compiler and
runtime correctness remain external interpretation boundaries.

The ordinary Real suffix theorem is an idealized mathematical interpretation,
not a claim that every FP32 reduction equals an unrounded sum. In particular,
nonfinite values can interact with the triangular matrix's zero coefficients.
Both variants retain exactly the same behavior. The optional native-graph
helpers retain the earlier inspected accumulating-dot model, but no accumulating
dot/separate-add rewrite or floating reassociation is assumed by the public
comparison theorem.

## Original defects and repaired domain

The original unspecified boundary load permits nonzero padded lanes to enter
active suffixes. Explicit zero-filled masked loads in the repair remove that
source defect. Another defect occurs for contiguous singleton layouts:
shape `(2,1,17,33)` with strides `(561,999999,33,1)` is contiguous, but the
original use of `stride(1)` sends the second flattened row beyond storage.
The wrapper passes canonical active-coordinate strides `(T*S,S,1)` relative
to the actual input data pointer, which covers those admitted singleton views.
The existing controls exercise both defects. No source defect is reclassified
as a floating-point experiment.

The production kernel continues to use widened PIDs/loop indices and clamped
inactive coordinates for both independent tails. The original Python and all
historical proofs remain unchanged. The prior public whole-wrapper results
are preserved in
[Misc/OptionalHostCorrect.lean](Misc/OptionalHostCorrect.lean) and
[Misc/OptionalHostFPEquiv.lean](Misc/OptionalHostFPEquiv.lean), under separate
optional namespaces. Public kernel entries do not import those results or
local host composition. The extracted input and memory-binding proofs retain
their bodies.

## Validation

Targeted Lean 4.29.0 builds use `/tmp/veritile-lean-toolchain` with
`LEAN_NUM_THREADS=2`. Public kernel correctness, intended FP equivalence,
source identity and success certificates pass standard-base axiom audits;
the mathematical formula is audited independently of both source bodies.

```sh
python bench/repairs/tritonbench_g/reversed_cumsum/check_contract.py
python bench/repairs/tritonbench_g/reversed_cumsum/check_kernel_variants.py
```

The contract check passes 12 rejections, 17 admissions and 261 program
executions over all nine configurations, including offsets, singleton strides,
both tails, guard words and input stability. The variant check binds the exact
Python comparison change and compares 36 execution pairs bitwise, including
exceptional FP32 values and multiple chunks/features. These are CPU storage
AST executions with Torch dot and mocked CUDA metadata. No GPU execution or
native arithmetic-conformance result is claimed.

`Correct.lean` has one public IO correctness specification. `FPEquiv.lean`
has one public typed-source equivalence specification and its empty numerical
assumption audit. `Misc/TensorBridge.lean` holds the optional physical-memory and
Tensor/check consequences without importing either reader entry. Targeted
builds for this IO reorganization include the actual source-safety proof,
logical IO connection, both entries and TensorBridge; no CPU/GPU experiment or
independent replay was rerun for the formatting change.

## Raw Python function bodies

`Kernels.lean` quotes both complete selected functions from the existing
repaired Python file, checked independently by `jit-correspondence.json`.
The Python original and its existing candidate are unchanged. All explicit
casts, masks, zero padding, statement order and descending loops are copied.
Pointer element types and constexpr arguments are outside the copied body;
independent input/output `FloatDType` parameters retain all six cases, while
the existing host wrapper still requires FP32 storage.

The tuple assignment, both signed program IDs, triangular comparison, feature stride multiplication, separate dot/add, and carry update after the store are retained. The candidate changes only the existing comparison operand order. The
internal execution views are proved equal to the current quoted sources.
Signed program IDs are part of the actual state; proof-only observation
adapters do not add source statements. Nonnegative lane coordinates use the
existing mathematical index projection, with native width/storage obligations
kept in the existing native binding and index refinements. These integer
connections assume no floating-point identity.
