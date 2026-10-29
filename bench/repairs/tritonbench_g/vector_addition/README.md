# vector_addition: FP32 kernel correctness and equivalence

The independent mathematical target is `out[i] = x[i] + y[i]`.
`Correct.add_kernel_correctness` uses the shared regional IO judgment to prove one program's canonical-source region
execution succeeds, writes the formula at active lanes, and preserves every
region cell outside its active output. It requires only active entry reads;
it imposes no separation between the two read-only inputs. The block size is
1024. `Proofs/Regional.lean::Regional.add_kernel_access_safe` proves active access bounds from the
region extents. The public acceptance scope is the kernel; the Python wrapper
supplies input assertions. Whole-wrapper allocation, returned-Tensor identity
and launch completion are not acceptance requirements.

## Input assertions and source

Both inputs must be strided FP32 CUDA tensors on the same device with the same
shape, default contiguous layout and no unresolved negative view bit. Their
common element count must be below `2**31`. Nonempty data pointers must be
four-byte aligned; empty inputs do not query their pointers. Scalar tensors and multidimensional contiguous tensors are admitted.
Contiguous offset views, input aliases, NaN/Inf, signed zeros and subnormals
remain admitted. No numerical transformation premise is a Python guard.
`Contract.lean` matches the existing guard order, and
`TensorBridge.python_checks_iff_preconditions` in optional `Misc/TensorBridge.lean`
proves exact admission equivalence.

The repaired Python retains the original `add_kernel` kernel AST and all
wrapper checks. Its readable `add_commuted_kernel` candidate changes only
`output = x + y` to `output = y + x`. The checked wrapper still launches the original kernel.
The original benchmark files are unchanged.

`Kernels.lean` contains both readable Triton DSL sources with typed FP32 regions.
Their explicit load, arithmetic and store tags remain in FP execution;
`.real` is the structural interpreter's opaque carrier. The public region
statement directly executes `originalKernel n`; the FP theorem executes that
same source. `Proofs/KernelChecks.lean` also verifies the optional flat IO and
FP source identities. `Proofs/Projection.lean` connects the source's Real
projection to existing mathematical proof helpers; projection does not justify
any FP transformation. The independent `formula xs ys i = xs i + ys i` is in
`Contract.lean`, separate from the source definitions.

`TensorBridge.checked_tensor_reads` binds both inputs through the same physical
reader using the core FP32 Tensor laws, checked contiguity and absent negative
view bits. It allows identical inputs and overlapping offset views, without
assuming different storage objects or disjoint read intervals.
`TensorBridge.checked_tensor_pointers` shows those views retain the actual launch
data pointers. These input binding laws do not assume a kernel result.
These laws identify the input views independently. The Real region theorem
takes their mathematical entry observations as its preconditions. This local
input binding does not prove Torch allocation or launch completion.

`Flat.add_kernel_flat_correctness` retains the optional flat-memory
corollary. Its `MaskedKernelIO₂.Implements` statement requires pairwise disjoint
x, y and output allocations. That stronger flat bridge does not describe all
Python-admitted inputs; read-only aliases are covered by the public region
result and independent Tensor read bindings above. No new Python assertion
narrows the admitted domain.

## Floating-point transformation

`FPEquiv.add_commuted_equiv` compares the actual original and commuted sources.
It requires both complete executions to succeed and their entire typed memory
functions to agree. Its domain is evaluated after the actual original load
prefix using the source store mask. Only active lanes require both actual
`M.fp32Load` results to be finite. Unspecified inactive load values are
unrestricted, including NaN/Inf. The memory proof splits active stores from
inactive lanes; it does not assume masked arithmetic values are equal.

The vector proof reuses the generic add_example rewrite proof through checked source identities. The generic `add_example/Proofs/MaskedDomain.lean` checks arbitrary inactive values,
empty domains, one-lane tails and rejection of either active nonfinite operand.

The public statement is `original n ≡[[]] commuted n`: addition commutation
requires no empirical rule. `ScalarArithmetic.add_comm_strict` connects the
shared `ScalarOpsLaws.add_comm` proof to the fixed `Guarded.fp32Theory`, whose
arithmetic profile is `Software.strictFP32` with `.ieeeValue`. The original
finite rewrite-site domain and all typed source operations remain unchanged.
`#print_fp_assumptions` distinguishes this strict model proof from empirical
assumptions; its empirical list is empty.

The historical `ReportedAdmission.fp32_add_commute` record at
`experiments/floating_point/report/summary.json#fp32/ADD-COMMUTE` remains intact:
ACCEPT, both gates PASS and 4096 replicates. It is no longer a premise of this
commutation theorem. This update neither replays nor extends the report.
Native primitive conformance remains a model boundary.

## Validation and preserved helpers

Run `python bench/repairs/tritonbench_g/vector_addition/Proofs/check_contract.py`. The check binds the
unchanged original AST, exact commuted AST and unchanged guards; it passes
17 rejection and 14 admission cases, plus 18 paired CPU kernel cases
covering empty/full/tail masks and NaN/+Inf/-Inf inactive loads. CUDA calls
are mocked; these checks are not GPU numerical evidence.

Offline CUDA lowering is inspected with
`python bench/repairs/tritonbench_g/add_example/check_cuda_lowering.py --task vector_addition --output-dir /tmp/vector_addition-kernel-lowering`.
Both variants target SM80/SM90 at FP32 and the selected block size. This checks
compiler output, not compiler correctness or GPU execution.

Targeted Lean 4.29.0 builds cover the public entries, shared-source identities,
active-mask domain checks and preserved helper compatibility. Standard-base
axiom audits exclude `sorry` and new axioms. Existing `Proofs/Real`, finite
memory, host and Tensor helpers remain optional historical results; they are
not imported by the public kernel entries. Earlier physical/lowering reports
remain historical evidence with their original scope.

## Public IO format

`IO.lean::Interface.addIO`, re-exported as `Correct.addIO`, declares the shared source, both input windows, output window
and active mask. `add_kernel_correctness` uses the shared `⊨ᵣ` regional judgment;
it includes actual source access safety, successful execution, logical addition
and the exterior raw-cell frame. It has no input-disjointness premise.
`Proofs/Flat.lean::Flat.add_kernel_flat_correctness` retains the optional disjoint
flat-allocation result as an ordinary theorem. Correct has one primary specification.
The original raw execution and access lemmas are retained as ordinary theorems
in `Proofs/Regional.lean`; they do not import the public reader entries.

`Misc/SourceBinding.lean` retains the source-identity helpers and the optional
`Rules`/`source` compatibility abbreviations under their existing `FPEquiv` names.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [SourceBinding.lean](Misc/SourceBinding.lean), [PublicChecks.lean](Misc/PublicChecks.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.
