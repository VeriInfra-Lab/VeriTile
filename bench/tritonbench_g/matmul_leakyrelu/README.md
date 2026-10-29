# matmul_leakyrelu

The current explicit admission revision is documented in the [checked wrapper](../../repairs/tritonbench_g/matmul_leakyrelu/README.md). It reuses the complete results below without changing the original Python.

The selected production profile has FP32 inputs and accumulation, the original
unspecified `tl.dot` options, 32×32×32 blocks, group size 4, and mandatory FP16
output. Real is the mathematical specification. The source does not constrain
input dtype, so the primary floating profile is FP32 under the source-selected
policy; test-only FP16 inputs do not create another required production profile.
Historical other-storage proofs remain supporting results.

## Primary whole-program entries

`Correct.lean` exposes
`Whole.Contracted.matmul_leakyrelu_contracted_mathematical_host_correctness`
with the `VeriTile.Spec.Real` marker. It proves the independent finite-K matrix
product and optional exact 1/100 leaky activation, followed by an explicit
`roundHalf`. The ideal primitive interface is constructed and proved.
`roundHalf := id` yields the exact mathematical output; IEEE FP16 narrowing
requires its separate numerical binding.

`FPEquiv.lean` exposes
`Whole.Contracted.matmul_leakyrelu_contracted_fp32_host_correctness`.
Every returned FP16 cell equals an independent logical-matrix recurrence with
an incoming live accumulator at each block dot, optional right-order slope,
and final FP16 conversion. The target has no memory, PID, registers, or source
execution and passes `#specNonCircular`. Six leaf `PrimitiveLaws` remain explicit:
initial zero, masked zero, dot-with-accumulator, ordered comparison, multiplication
by the right-hand slope literal, and FP16 conversion. Both activation classes
retain all six premises, including K=0 and empty-grid calls.

Both entries use `Whole.contractedKernel`, which directly reuses
`FPFused.source ... .single`. `Proofs/ContractedWhole.lean` connects its existing
per-program proof to every PID on evolving memory, the complete grouped grid,
host assertions/return, and an exact full memory frame. The module proves that
A/B/C offsets and active-store masks are identical to the original geometry;
`Proofs/WholeTask.lean` and `Proofs/FiniteDomain.lean` supply the shared coverage,
unique stores, finite arithmetic, allocation spans and address proofs.

The original separate-add `Whole.matmul_leakyrelu_mathematical_host_correctness`
and `Whole.matmul_leakyrelu_fp32_host_correctness` declarations are preserved as
supporting source-model results. Their floating graph is distinct from the
contracted graph. They are not the primary native-profile registration.

## Source and native correspondence

The pinned original `matmul_kernel` and `leaky_relu` function/decorator ASTs were
compiled unchanged for FP32 A/B pointers, an FP16 C pointer, dynamic i32 dimensions
and strides, fixed32/G4, CUDA SM90, and Triton 3.7.1. The default options resolve
to TF32 input precision, fusion enabled, 4 warps and 3 stages.

Both activation cases emit `tt.dot(a,b,live_accumulator)` without a separate
floating add. The activated graph compares the accumulator with zero, multiplies
`accumulator * 0.01f` (literal bits `0x3c23d70a`), selects, then narrows to FP16.
The Lean contracted model follows this observed graph directly. The floating
primitive is not replaced with a Real sum or a separate dot/add recurrence.

Six offline cases cover activation off/on with fusion omitted/false/true; 36
compiler artifacts were checked. The two omitted-fusion cases bind the primary
profile; the other four provide comparisons. This is evidence for the pinned
signature, target and toolchain. Other targets, specializations or compiler
settings need their own correspondence evidence. Ordinary correct compiler
lowering and successful runtime execution of that observed graph remain trust
boundaries. No compiler implementation proof or GPU execution is claimed.

`FPGraphDifference` refutes unconditional source/contracted floating equivalence.
No such equality is asserted, assumed, or deferred as a true numerical rule.
The ideal Real projections coincide without implying native floating equality.

## Domains and covered behavior

The public `TensorContract` binds actual rank-two metadata, FP32 A/B, FP16 C,
compatible devices, contiguous A, admitted nonnegative B strides, the actual
contiguous fresh `torch.empty` output, aligned data pointers, complete allocation
spans, and consistent read-only input aliases. A/B may physically alias; C is
physically disjoint from their spans. Initial output contents are arbitrary.
Torch metadata/allocation and a successful compatible launch are ordinary API
contracts; no callback supplies a kernel result.

`IndexDomain` is a visible sufficient finite domain. Signed-32 controls,
dimensions, strides and all formed element offsets stay below 2^31, including
padded lanes and final K pointer advances. Formed byte pointers stay below 2^64;
active 4/4/2-byte accesses fit the actual allocation spans. Every logical output
has exactly one writer. Negative strides and overflow-sized inputs are outside
this domain.

The full `ceil(K/32)` loop, K padding, small/ragged M/N, shortened final row group,
and transposed/sliced B views are covered. Only `activation == "leaky_relu"`
activates the leaky branch; all other strings select identity. M=0 or N=0 returns
the fresh empty allocation without running a program. K=0 retains the zero,
activation and cast operations. Dimension failure precedes the contiguous-A
assertion; there is no B-contiguity assertion.

## Validation and remaining numerical work

Six exact Lean modules compile with module axiom audits. Existing source evidence
contains 18 unchanged-original CPU interpreter value cases and two original
assertion failures, plus a source/domain audit. These CPU cases validate source
control/layout behavior and do not establish native TF32 numerical conformance.
The native evidence records six successful offline compilations and hashes.

The exact numerical registration has six rule families, five primitive ports,
two activation cases, and twelve bindings. TF32 operand conversion and MMA
accumulation, rounding, NaNs/Infs, signed zero, FTZ, the slope literal/product,
and FP16 conversion remain explicitly deferred. No GPU result comparison or
bitwise conformance is claimed. Root review owns installation, registration,
and the canonical whole-task acceptance count.
