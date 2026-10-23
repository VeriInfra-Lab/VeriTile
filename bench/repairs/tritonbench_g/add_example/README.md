# add_example: FP32 kernel correctness and equivalence

The independent mathematical target is `out[i] = x[i] + y[i]`.
`Correct.add_kernel_correctness` proves one program's canonical-source region
execution succeeds, writes the formula at active lanes, and preserves every
region cell outside its active output. It requires only active entry reads;
it imposes no separation between the two read-only inputs. The block size is
4. The same public IO judgment proves active source access safety from the
declared region windows. The public acceptance scope is the kernel; the Python wrapper
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
`TensorBridge.python_checks_iff_preconditions` proves exact admission equivalence.

The repaired Python retains the original `add_kernel` kernel AST and all
wrapper checks. Its readable `add_commuted_kernel` candidate changes only
`output = x + y` to `output = y + x`. The checked wrapper still launches the original kernel.
The original benchmark files are unchanged.

`Kernels.lean` contains both readable Triton DSL sources with typed FP32 regions.
Their explicit load, arithmetic and store tags remain in FP execution;
`.real` is the structural interpreter's opaque carrier. The public `Correct.addIO` exposes the source, buffers, windows and mask.
Its shared `MaskedKernelIO₂.RegionalImplements` judgment executes `originalKernel n`; the FP theorem executes that
same source. `Misc/InterfaceChecks.lean` also verifies the optional flat IO and
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

`KernelResults.add_kernel_flat_correctness` retains the optional flat-memory
corollary in `Proofs/KernelResults.lean`. Its `MaskedKernelIO₂.Implements` statement requires pairwise disjoint
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

`Proofs/MaskedDomain.lean` checks arbitrary inactive values,
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

Run `python bench/repairs/tritonbench_g/add_example/check_contract.py`. The check binds the
unchanged original AST, exact commuted AST and unchanged guards; it passes
17 rejection and 11 admission cases, plus 18 paired CPU kernel cases
covering empty/full/tail masks and NaN/+Inf/-Inf inactive loads. CUDA calls
are mocked; these checks are not GPU numerical evidence.

Offline CUDA lowering is inspected with
`python bench/repairs/tritonbench_g/add_example/check_cuda_lowering.py --task add_example --output-dir /tmp/add_example-kernel-lowering`.
Both variants target SM80/SM90 at FP32 and the selected block size. This checks
compiler output, not compiler correctness or GPU execution.

Targeted Lean 4.29.0 builds cover the public entries, shared-source identities,
active-mask domain checks and preserved helper compatibility. Standard-base
axiom audits exclude `sorry` and new axioms. Existing `Proofs/Real`, finite
memory, host and Tensor helpers remain optional historical results; they are
not imported by the public kernel entries. Earlier physical/lowering reports
remain historical evidence with their original scope.

`Proofs/KernelResults.lean` preserves the earlier raw regional, flat and separate
access-safety corollaries as ordinary theorems. `Misc/InterfaceChecks.lean` checks the
reader-facing source identities; supporting proof modules do not import readers.

`Misc/SourceBinding.lean` retains the source-identity helpers under their existing `FPEquiv` names.

## Optional proof files (2026-10-21)

The optional native, Tensor, audit and application entries for this selected revision are in `Misc`:

- `Misc/InterfaceChecks.lean`
- `Misc/SourceBinding.lean`
- `Misc/TensorBridge.lean`

The primary Contract/Correct/FPEquiv entries and their required IO, source and Proofs dependencies keep their existing statements and scope. This layout change adds no mathematical or numerical claims.
