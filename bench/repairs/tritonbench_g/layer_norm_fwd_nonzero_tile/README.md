# Positive-tile layer-normalization repair

Current FP entry (2026-10-21): `FPEquiv.lean`'s `index_commuted_equiv` proves the actual
`original ≡[[]] indexCommuted` pair over the complete ordinary grid. The
candidate is the existing FP32 source: only X's initial integer
address product changes from `row * stride.x` to `stride.x * row`. Both sides
use the selected positive `tile N`, including N=0. The conclusion includes
successful actual row executions, source access safety, all seven typed output
channels and both exact exterior frames. No numerical rule or experiment is
needed for this integer-index transformation.

The candidate changes the wrapper's tile expression to
`max(1, min(MAX_FUSED_SIZE, triton.next_power_of_2(N)))`. The input guard,
allocation order, launch arguments, decorators, and complete JIT body are
unchanged. Zero columns remain admitted. Nonzero widths select exactly the
original tile; widths above 16384 remain rejected after output allocation and
before launch.

The original repair's admitted `(M, N) = (2, 0)` case selects `BLOCK_N = 0`.
The installed Triton 3.7.1 offline compiler rejects its `tl.arange(0, 0)`.
The separate frozen diagnostic is
`.veritile-cache/layer-norm-fwd-zero-width-diagnostic/report.json`; it also
contains the successfully compiled unchanged-body `BLOCK_N = 1` control.
`check_candidate.py` verifies this control's artifact hashes, the exact sole
AST change, and the installed launch binder. It makes no GPU numerical claim.
The original source and proof files are retained unchanged.

The standard entry modules have these obligations:

| Module | Result |
| --- | --- |
| `Contract.lean` | Reuses the exact input admission checker; proves tile positivity, the arange domain, unchanged nonzero tiling, zero admission, and oversized rejection. |
| `Kernels.lean` and `Source.lean` | Retain the entire typed source, prove its identity with the selected positive-tile kernel, and execute every row without a success premise. |
| `RealIO.lean` and `NativeIO.lean` | Expose the selected source, all finite input/output channels, strides, branch masks, ordinary-grid schedule and independent formulas. |
| `Correct.lean` | One public `specification`: the complete ordinary grid implements the independent Real formula through shared `AlgorithmProgramIO.Implements`. |
| `FPEquiv.lean` | One public `specification`: the original and existing index-commuted source grids are equivalent with an empty experimental rule set. |
| `Misc/NativeCorrect.lean` | Ordinary theorem: the complete ordinary source grid implements its ordered primitive reference through shared `OrdinaryProgramIO.Implements`. |
| `Misc/RawRealResults.lean` | Preserves the original full `WholeGrid.Post`, all eight return observations and the zero-column Real statistics theorem. |
| `Misc/RawNativeResults.lean` | Preserves the full tagged source observation/frame, actual compiler-artifact/current-memory native execution, exact native statistics and physical store results. |
| `Misc/TensorBridge.lean` | Applies those results to physical memory and Tensor byte reads, including requested-width bytes and their exterior frames. |

The public judgments use nine finite input channels and seven writable output
channels: `Y`, `Y1`, `Mean`, `Rstd`, `RESIDUAL_OUT`, and the two dropout-mask
streams. Their windows retain the independent source strides and enabled-branch
masks. Each row executes the original source from the same initial state, with
source-derived disjoint writes; the interface does not replace this grid with a
serial execution. Both judgments prove actual source load/store safety from the
stated channel capacities. Success, output values and the exact exterior frame
hold independently of those capacity bounds.

The preserved eight-field return observations also cover unchanged seeds and
the residual fallback input view. Input aliases, arbitrary source cells with
typed decoded defaults, zero rows/columns and inactive output strides retain
their original domains. `Proofs/RealImplementation.lean` reuses the original
`RealRevision.ActiveGrid.source_correct`; `Proofs/NativeImplementation.lean`
reuses `TypedSource.source_grid_post`. The raw and Tensor results use these same
execution foundations. Input/output bindings, ordinary-grid transport and source
trace safety are ordinary helpers, keeping both public readers short.

The source primitive result in `Misc/NativeCorrect.lean` is generic over `Algebra` and its
explicit comparison interface. It remains an optional single-source theorem,
separate from the pair in `FPEquiv.lean`. The comparison interface only supplies
the source's opaque comparison operation; it asserts no order or Real law.
The compiler-artifact and requested-storage native
claims remain in `Misc/RawNativeResults.lean`; no new numerical assumption or
experiment is introduced by this format adaptation.

`Source.indexCommuted` in `Kernels.lean` instantiates the existing benchmark candidate
at the same FP32 source profile. `Proofs/Pair.lean` reuses the benchmark's
complete-source `FPProof.whole_state` proof at the selected `tile N`, including
zero columns and all optional flags. Its empty-rule source equivalence is
proved and built. `Proofs/PairSafety.lean` separately proves the changed pure
prefix safe, transports its equal actual successor state, and reuses the
identical complete suffix. `Proofs/GridPair.lean` transports every successful
ordinary row and the same disjoint memory merge. The task-local syntax in
`Proofs/OrdinaryEquivalence.lean` retains actual invocations, row write sets and
observers as program data; its fixed structural relation requires both full
successful ordinary runs, both safety conclusions, all active typed output
cells and both frames. It uses no custom executor or correctness callback.
The input domain, width checker, separation and geometry conditions are
unchanged, with no positive-row/column restriction or output/success premise.

The independent Real entry is unchanged. Its finite input arrays feed the
mathematical `Formula` definitions for both Philox streams, optional residual,
column means and squared deviations, `Real.sqrt`, and both affine outputs.
`Proofs/RealImplementation.lean` connects the source grid to those definitions;
the Real formula does not refer to source execution or native observations.

At `N = 0`, every launched row still stores `Rstd`; layer-normalization mode
also stores `Mean`. Column stores to `Y`, `Y1`, `RESIDUAL_OUT`, and the dropout
mask are masked off. `zero_columns_writes` and `TensorBridge.fp_zero_columns_frame` state
this exact distinction. `fp_zero_statistics` retains division by the native
scalar zero and the ordered zero-leaf variance tree. Its mean numerator
retains the padded stream's optional rowscale/dropout operations. No NaN-bit
pattern is invented, and no native equality with the total Real convention
`mean = variance = 0`, `rstd = 1 / sqrt eps` is asserted.

The native boundary is explicitly local. `Compiler.Artifact` selects the
owner-specific reduction trees and affine contractions and recursively binds
individual opcodes, ordered operands, source sites, store masks, and store
targets. `StorageCompiler.Artifact` adds each requested destination conversion.
These bindings are the declared compiler interpretation; this packet does not
verify the installed compiler implementation. No artifact field contains a
final memory, tensor-output equation, or execution-success premise.

`Primitives` gives total native FP32 operations, Philox conversion, the host
binary64-to-FP32 scalar ABI conversion, and requested storage casts. Its carrier
retains nonfinite values and signed zero and can represent all requested output
encodings. Mean/rstd stores remain FP32, seeds use int64 cells, and masks use
Boolean cells. The model does not replace source arithmetic with exact Real
arithmetic or require a numerical experiment to justify the primitive boundary.

`StorageMachine.execute` evaluates each enabled operand from current memory,
applies its destination cast, and performs source-order stores. The proof that
each earlier store preserves the input reads derives the independent snapshot
reference. `TensorBridge.fp_tensor_output` composes that reference with Tensor pointers,
strides, coordinate bounds, and requested element widths, without assuming a
final-value equality. `TensorBridge.fp_tensor_input` preserves the Tensor bytes of returned
seeds and the residual fallback. `ByteExecution` realizes those same scalar stores on their
local byte spans and frames every exterior byte. Its descriptor conditions are
kernel-entry facts; allocator, host lifecycle, device-resource, and launch-grid
product verification are outside this packet.

The candidate-local typed chain avoids an observed import ABI collision:
the old cached FP environment and `VeriTile.Triton.Float.Reciprocal` both contain
`VeriTile.Triton.FP.Structural.evalExpr.eq_4`. The failure is recorded in
`.veritile-cache/kernel-evidence/layer_norm_fwd_nonzero_tile/source-build-01.log`.
The local chain reuses valid source proofs and removes the unnecessary legacy
FP import; it does not alter the original files or weaken theorem statements.

Validation evidence and the frozen source/dependency/build-receipt manifest
are under `.veritile-cache/kernel-evidence/layer_norm_fwd_nonzero_tile/`.
The scoped build helper is
`.veritile-cache/build-tools/workspace_lean_local_first_v2.py`.

Current public-IO format validation is recorded separately under
`.veritile-cache/public-io-format-20261020/layer_norm_fwd/`. It leaves the frozen
historical evidence unchanged and makes no independent replay or GPU claim.

The completed pair's targeted builds, current main/auxiliary import checks and
source/object receipts are recorded under
`.veritile-cache/two-task-completion-20261021/layer_norm_fwd/`. Existing dependency
objects are reused by the strict scoped builder; this adds no host, per-task
FP32-correctness, GPU or numerical-experiment acceptance requirement.

Optional proof layout (2026-10-21): `Misc/NativeCorrect.lean`, `Misc/TensorBridge.lean` are optional entries in `Misc/`. The required `Contract.lean`, `Correct.lean`, `FPEquiv.lean` and their proof helpers do not import these entries. Declaration names, statements, proof terms and acceptance status are retained; this move adds no mathematical completion or numerical experiment.

Additional optional reader paths: `Misc/RawRealResults.lean`, `Misc/RawNativeResults.lean`. Their declaration bodies are retained. Historical readers remain separate from current primary/auxiliary checks.

Source readability maintenance (2026-10-22): `Kernels.lean` now contains the
complete ordinary Triton quotation for each selected FP32 program, including
all optional streams and stores. The selected original and index-commuted
candidate are definitionally equal to the previous shared macro instantiations
for every flag, region, stride, size, scalar, tile and arange-domain witness.
The source-only campaign cache records that complete function equality and
fresh joint public/retained optional checks. No upstream source, domain or
proof acceptance requirement changed.
