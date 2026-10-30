# `rbe_triton_transform_checked_launch` candidate

This separate candidate closes the launch guard gap in the currently selected
`rbe_triton_transform_repaired.py`. The original canonical source and frozen
22-module conditional proof packet are unchanged. The selected checked revision
retains this candidate's source and guarded domain.

The original guards admit nonempty shape `(1,131071,1)`, position zero, but the
actual grid is `(1,65536)`, above the supported CUDA grid-y limit of 65535. The
original native theorem requires `Index.SupportedLaunch` separately, so that
packet establishes conditional kernel correctness, not whole-domain acceptance
of the original Python wrapper.

The candidate adds only two guards after the original position checks. For a
nonempty input it requires `batch <= 2^31-1` and
`ceil(M/2) * ceil(K/1024) <= 65535`. Every empty shape keeps the original domain.
All prior Tensor, dtype, layout, device, contiguity, lazy-negative, alignment,
position and padded-position conditions remain unchanged. The helper, kernel,
allocation, strides, grid expression and launch call are textually unchanged.
There is no copying or grid splitting.

`Contract.checkInputs_iff` connects the candidate checks to its exact contract;
`Contract.domain_exact` states the domain restriction, and
`Contract.empty_domain_unchanged` proves that empty admission is unchanged.
`Input.supported` derives the original launch premise from the same Tensor's
shape and the new checks.

The Real endpoint is `Correct.kernel_correct`; the ordinary
`FPEquiv.kernel_native_equivalence` theorem is retained in `Misc/NativeCorrect.lean`.
They use the shared `ProgramIO.Implements` judgment on independent logical
Real/native arrays and reuse the original successful DSL execution. The
complete Tensor bindings, byte stores and native address statements are retained
in `Misc/TensorBridge.lean`; its candidate application has no separate launch-support
premise. Scalar conversion/trig/FMA/compiler bindings
remain the original explicit model boundaries; this candidate adds no numerical
assumption, experiment or GPU execution.

The static audit checks that deleting exactly the two new guard statements
recovers the pinned original Python file byte-for-byte. Its metadata-only guard
prefix checks exercise both launch boundaries, the original counterexample and
unchanged empty admission without importing torch or launching a kernel.

`IO.lean` is the IO descriptor shared by the Real and optional native results. It lists one
FP32 input and one FP32 output, both spanning `[0, request.size)`, no scratch,
the exact selected source, and the original complete two-axis schedule. Every
invocation clears local registers and installs the original program IDs. The
Real entry uses identity interpretation; the native entry exposes word input
encoding and output-word observation under the existing local primitive/FMA
profile. It introduces no transformation law or numerical experiment.

`Proofs/AccessSafety.lean` checks the actual selected source's active real and
imaginary loads/stores; odd missing partners stay masked. `Proofs/TypedGrid.lean`
connects the existing complete-grid proof to exact FP32 output cells, and
`Proofs/IOCorrectness.lean` composes these into source safety, successful execution,
every independent logical output and the full exterior cell frame. This regional
IO judgment does not itself claim flat-byte pointer transport.

`Proofs/RawResults.lean` retains the previous unrestricted raw-state Real/native
results. `Misc/TensorBridge.lean` preserves the checked Tensor applications,
signed-index bounds, native address correspondence and complete physical frames
as ordinary theorems. Its prior physical backing and primitive model boundaries
are unchanged. Targeted Lean builds cover all new and edited modules plus the
Tensor and audit consumers; no independent replay or GPU run is claimed.


## Complete mask reuse pair (2026-10-21)

`FPEquiv.mask_reuse_equiv` compares the actual complete original source against
`maskReuse`, through the shared `ProgramIO` implementation-pair judgment.
The new `rbe_triton_transform_mask_reuse.py` preserves the checked wrapper and
frequency helper and changes only `out_real_mask = x_real_mask` and
`out_imag_mask = x_imag_mask`. Reusing the load masks removes the duplicate
bounds comparisons before the two stores; no performance measurement is claimed.
The source coordinates and bounds are unchanged between loads and stores.

`Kernels.lean` now contains both complete ordinary `triton` blocks. It writes
out the already accepted mathematical index projection instead of invoking a
whole-source adapter. Signed token values remain `Int`, nonnegative controls
remain `Nat`, and the implicit FP32 pointer-element store casts are explicit.
`Misc/SourcePreservation.lean` proves complete `ComputeKernel` equality with the
former selected source, including ports and every precision scope. The main
source file imports no proof or optional module.

`Proofs/Pair.lean` proves full-state equality and actual trace-safety equivalence
for arbitrary floating payloads and lifts them to every original two-axis launch.
The main pair retains all dimensions, odd tails, empty shapes, signed positions,
the original arbitrary complete legal schedule, FP32 input/output channels and
full exterior frames. All floating calculations, conversion tags and barriers
are unchanged, so the experimental rule set is empty. The generic scalar backend
and fixed input/output interpretation admit the retained all-word native view;
its existing trig/FMA/compiler profile remains an optional separate application.
The independent Real `Correct.kernel_correct` is unchanged.

Optional reader and application modules are in `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean), [Misc/Audit.lean](Misc/Audit.lean), [Misc/SourcePreservation.lean](Misc/SourcePreservation.lean). Their declaration namespaces and proof statements are preserved. The main `Contract.lean`, `Correct.lean` and `FPEquiv.lean` entries remain at the task root.

`IO.lean` holds the single `rotationIO` layout and both original/candidate invocation and program-view constructors. `Proofs/Pair.lean` retains execution, access-safety and full-grid equivalence proofs.
