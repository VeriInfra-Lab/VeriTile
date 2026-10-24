Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

# Checked RMSNorm kernel

The selected implementation is `rms_norm_triton_repaired.py`. Its JIT body is
identical to the original benchmark. The existing wrapper checks select FP32
CUDA tensors, matching normalized/weight shapes with positive normalization
length, a finite positive Python-float epsilon, absent lazy flags, and the
source's integer/pointer envelope. Both original Python and repair Python are
unchanged by this proof completion. Original inputs may be strided, broadcast,
offset views, or share storage: the source intentionally calls `contiguous()`
on both of them.

The primary endpoints are `Correct.kernel_correct` and
`FPEquiv.index_commuted_equiv`. The former implements the independent Real
formula; the latter compares the actual original and existing index-commuted
candidate as `original ≡[[]] indexCommuted`. `IO.lean` fixes the actual source,
[M,N] input/output channels, [N] weights, row-major windows, complete row order,
FP32 graph interpretation and native observer. `Proofs/Pair.lean` proves source
access safety, successful execution, equal observed cells and both exterior
frames for every complete row permutation. X and W may alias each other;
both remain separate from Y. All symbolic input words, runtime epsilon words
and lane-specific reduction profiles retain their original domain.
`Misc/NativeCorrect.lean` retains the complete ordinary native recurrence theorem
under its old full name `FPEquiv.kernel_native_equivalence`.
`TensorBridge.Real.kernel_correct` and
`TensorBridge.Native.kernel_native_equivalence` retain the complete checked
Tensor applications, including physical readback and all-memory equality.
The bridge connects finite indices and padded pointers to the actual storage
bases and capacities. Empty row grids leave memory unchanged, and unused
empty backing storage requires no live pointer.

`Contract.lean` defines the checked input contract and independent mathematical
result:

`x[row,i] / sqrt(sum_j x[row,j]^2 / N + eps) * weight[i]`.

The sum ranges over exactly the logical normalization length. Actual Tensor
objects use the shared `VeriTile.Torch.FP32.Tensor` and `TensorLaws`, together
with their queried strides. Logical input values come from the same backing
allocation and `physicalIndex` map. The ordinary contiguous-copy preservation
law is used only to connect those input values to the actual kernel loads.
It supplies no RMSNorm result.

`Kernels.lean` shares the readable typed DSL in
`bench/tritonbench_g/rms_norm_triton/Kernels.lean`; `source_identity` and
`source_row_identity` bind the selected row kernel to that source. The DSL
retains the FP32 input cast, zero-padded masked loads, runtime-epsilon ABI
annotation, reciprocal of sqrt, and output cast **before** weighting. The
native auxiliary theorem compares execution against an independent native
instruction recurrence. The pair reuses the already specified
`Proofs/LegacyKernels.typed .fp32 .fp32 Bool.true` candidate: its row bases use
`N * pid` in place of `pid * N` for both X and Y. The original is
definitionally the existing `Bool.false` source. The proof uses natural-number
multiplication commutativity and retains every floating operation, cast,
reduction, mask, store and observer. No new Python candidate is introduced.

Native scalar IEEE operations, literal/integer conversion, runtime epsilon
conversion and the compiler's actual lane-specific reduction trees are
explicit model boundaries. Reduction trees preserve every padded lane and
allow native FMA contractions. `Observed.reference_singleton` proves that
`N = BLOCK_SIZE = 1` retains the fused `x*x + runtime_epsilon` case. The actual
FP32 epsilon word is bound separately from its Real annotation, including
underflow/overflow of Python-to-FP32 conversion. No exact-real identity is
assumed for rounded operations, and no numerical experiment is used to admit
a primitive or compiler binding.

The historical wrapper-outcome/device-context proofs remain optional in
`Misc/HostCorrect.lean` and `Misc/HostFPEquiv.lean`. The public dependency
chain does not execute a runtime, prove allocation internals, or claim a GPU
conformance result. The original benchmark's missing checks and possible
signed row-offset overflow remain preserved in its unchanged Python source.
Historical 16-dtype-pair slice/address-equivalence helpers remain under
the original directory's `Proofs/Legacy*`; this pair selects the existing FP32
branch and connects it to the complete selected schedule.

Validation: targeted Lean builds of the source, IO, pair bridge, native
auxiliary, two public endpoints and Tensor bridge; checked
non-circular specifications and standard-base axiom audits; AST comparison of
the original/repair JIT and all existing guard expressions; and the existing
fixed-shift block-size source controls. No new numerical experiments or GPU
runs are claimed.

## Kernel and Tensor interfaces

`Correct.kernel_correct` states the complete Real RMSNorm grid as a shared IO
implementation of the independent logical formula.
`FPEquiv.index_commuted_equiv` uses actual implementation views of the original
and candidate schedules, lifted from their full `ProgramIO.Equivalent` proof.
The retained ordinary `FPEquiv.kernel_native_equivalence` theorem uses
`ProgramIO.Implements` for the independent native recurrence. Their encoder is
`Term.input` and their observer retains the original per-lane reduction
topology. The runtime epsilon word and primitive instruction ports remain
explicit. The pair has no floating transformation assumption or numerical
experiment dependency.

`Proofs/SourceSafety.lean` proves the actual row trace safe and composes every
row in the exact schedule. `Proofs/PublicIO.lean` binds logical inputs and
reuses the existing complete-grid result; it also preserves empty rows and
empty normalized axes. `Proofs/KernelReal.lean` and `Proofs/KernelObserved.lean`
retain the earlier raw operational statements as ordinary theorems.

The original complete checked Tensor applications are ordinary theorems `TensorBridge.Real.kernel_correct` and `TensorBridge.Native.kernel_native_equivalence`; their prepared-storage correspondence, physical memory results and supported domain are unchanged.

## Source-only organization (2026-10-22)

The original continues to reuse the complete public ordinary FP32 quotation in
`bench/tritonbench_g/rms_norm_triton/Kernels.lean`. The selected existing
`indexCommutedKernel` is now a complete ordinary quotation in the repair's
public `Kernels.lean`, preserving both integer row-base commutations and every
floating operation/cast. `Proofs/SourcePreservation.lean` retains the original
`legacy_original_identity` name and proves equality of both complete source
functions to the historical typed family by `rfl`; the pair imports that proof
entry. The historical generic precision branches and source checker are unchanged.
All original domains, runtime epsilon words, reduction profiles, grids, safety,
frames, main declarations, numerical assumptions and completion counts remain
unchanged.
