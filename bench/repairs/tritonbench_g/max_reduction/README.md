# max_reduction — FP32 kernel specifications

This revision proves the three kernels and their retained implementation
variants. Python supplies input assertions. Wrapper execution, allocation,
autotuning trials, launch orchestration and returned Tensor identity are not
required proof obligations. The immutable original remains in
`bench/tritonbench_g/max_reduction/max_reduction.py`.

## Mathematical targets and source precision

For contiguous logical input `x`, the first stage writes
`mid[p] = max { x[p*B+i] | 0 <= i < B, p*B+i < N }`. The second stage writes
the maximum of its active intermediate entries. Their mathematical IO
specifications use an independent finite supremum, with bottom for masked
padding. The abstract Real readback of an all-padding result is zero; admitted
nonempty production blocks have an attained finite maximum. This convention
does not identify hardware negative infinity with real zero.

For logical shape `[M,N,K]`, each active dimension row `(m,k)` writes
`v[m,k] = max { x[m*N*K+n*K+k] | 0 <= n < N }` and the least `n` attaining
that value. `Correct.dimension_math_correct` uses the shared `ProgramIO.Satisfies`
judgment to state attainment, upper bounds and the least tie index over logical
input arrays. `Spec.lean` defines this independent predicate, including the
retained zero-value/index fallback for an empty reduction axis. `IO.lean`
exposes the actual source and the separately typed value/index output channels.
`BN = nextPower N` covers the complete axis; ties and masked tails are included.

Input, intermediate storage, reduction values and value outputs are FP32.
The indexed reducer selects an int32 index and stores its signed-int64
extension. All actual index arithmetic is widened to int64 in the separate
Python repair; `Proofs/Machine.lean` proves the index/mask/pointer connection.
No extra floating dtype is selected. Real is the mathematical specification.

## Sources and implementation equivalence

`Kernels.lean` directly pastes the three complete, unchanged `@triton.jit`
function bodies from `max_reduction_repaired.py`. The three bodies in
`candidate_kernels.py` are newly written native counterparts of the existing
Lean integer/mask variants, not recovered historical Python. Every cast,
statement, paired reduction, mask and store remains in source order.
`jit-correspondence.json` checks all six complete bodies. Pointer storage,
annotated int32 runtime arguments and constexpr bindings are outside the
quotations; no Lean declarations or antiquotations are inserted into a body.
Reduction models remain in `Proofs/ReductionModel.lean`.

`Proofs/Source.lean` re-proves execution of these actual widened sources,
including both dimension output streams and complete memory effects.
`Proofs/AccessSafety.lean` checks their actual successor-state traces. The
address connection uses integer cast identities, without floating conversion
assumptions. Paired `tl.max(..., return_indices=True)` retains its FP32 value
result and the existing signed-int64 output-index conversion; lowering does
not introduce an extra FP32 value cast.

The pre-existing candidates are retained:

- Stage 1 commutes the integer product `pid * BLOCK_SIZE`.
- Stage 2 changes `offset < mid_size` to `not (mid_size <= offset)`.
- The dimension kernel commutes `pid_m * BLOCK_M`.

These are exact integer rewrites. Each public `FPEquiv` theorem displays the
two actual sources as `original ≡[[]] candidate` and proves successful
execution with equality of every typed memory cell for arbitrary program IDs.
`Misc/NativeCorrect.lean` retains the shared `ProgramIO.Equivalent` theorems,
including source trace safety, observed outputs and exterior frames. The same total reduction engine
is part of both program types. It can instantiate any selected backend tree,
including its FP32 NaNs, infinities, signed zeros and masked padding. No
floating reassociation, reduction-tree permutation or finite-value guard is
used. In particular, padding remains inside indexed native reductions: an
unordered comparison can select it, and the native model retains that behavior.

**No numerical transformation experiment is deferred for this revision.**
The old native-to-IEEE scalar conformance ports describe a primitive modeling
boundary; they are not per-task numerical transformation assumptions.

## Input contract and memory meaning

Both wrappers require strided FP32 CUDA input, no unresolved negative view,
fewer than `2**31` elements, and a 4-byte aligned actual data pointer when
nonempty. Global `max` also requires nonempty, contiguous input. The global
block width retains the exact integer ceiling-square-root calculation.

`max_dim` checks an `operator.index` dimension in `[-rank,rank)`, a reduction
axis length in `[1,32768]`, and `K <= 65535` when `M != 0`. The existing
`inp.contiguous()` belongs to the original source and is retained; the
revision adds no new input conversion or general layout compatibility. Empty
outer dimensions launch no kernel. The original `BLOCK_M` choices 8, 16, 32
and `BLOCK_N = nextPower N` remain.

`Contract.lean` and `Proofs/Contract.lean` mirror these assertions. The shared
`VeriTile/Torch/FP32.lean` API laws connect contiguity, negative-view flags and
the observed pointer to backed FP32 storage. `checked_contiguous_read` derives
direct-load/logical-element agreement from those laws. For `max_dim`, the
already existing Torch contiguous operation supplies the contiguous kernel
input and preserves logical values at the library boundary. The kernel
statements explicitly bind logical input values to memory, and flat IO
statements prove active address safety and allocation frames. Tensor API
sources are cited in the shared module; pointer width/alignment references
remain in `documents/GpuMemoryModel.md`.

## Original defects and separate repair

The original global wrapper does not enforce the contiguous assumption used
by its flat pointer arithmetic. For storage `[1,100,2,1000]`, the stride-2 view
has logical values `[1,2]`; two consecutive pointer loads see `[1,100]` and
produce 100 instead of 2. The separate repair rejects this layout explicitly.

The original dimension wrapper computes `K = inp.numel() // M // N`. Shape
`(0,3)` with `dim=1` gives `M=0` and divides by zero. The separate repair uses
the suffix shape product and skips empty outer grids. Neither defect is
counted as a proof of the unguarded original program.

## Proof organization and checks

`Correct.lean` contains three public specifications: the two existing flat IO
stage results, whose visible source/windows/mask fields are in `IO.lean`, and
the independent dimension least-argmax IO result. `FPEquiv.lean` contains the
three source-pair equivalences for the actual retained variants. The prior
source-safe shared IO results are ordinary theorems in `Misc/NativeCorrect.lean`.
The stage statements preserve their arbitrary program-id scope; they are not
substituted for the retained whole-grid results. `Misc/TensorBridge.lean` contains the optional Python-check correspondence
theorems `TensorBridge.python_global_checks` and
`TensorBridge.python_dimension_checks`. Supporting source execution, memory, integer and reduction proofs
live in `Proofs/`. `Proofs/SourceEquivalence.lean` and
`Proofs/LegacyRealResults.lean` preserve the former raw source results as
ordinary helper theorems. Actual source safety and logical IO connections are
in `Proofs/AccessSafety.lean`, `Proofs/PublicEquivalence.lean`,
`Proofs/DimensionIO.lean` and `Proofs/RealIO.lean`. The whole two-stage global
grid, complete dimension grid and historical host results are retained in
`Proofs/LegacyCorrect.lean`, `Proofs/LegacyFPEquiv.lean`, `Misc/ASTCorrect.lean` and
`Misc/ASTHost.lean`. `Misc/Audit.lean` builds these retained consumers alongside
the public entries and Tensor bridge. Existing attained-maximum target lemmas were moved to
`Proofs/KernelMath.lean` for shared reuse.

Targeted Lean builds audit public statement surfaces and theorem axioms against
the standard base. All three equivalences print an empty floating-assumption
report. The original Python repair and input contract are unchanged; the Lean sources now retain their original integer casts explicitly.
The following Python/lowering results are retained prior evidence: `python check_contract.py` passes 71 actual-wrapper/body cases, including
all block choices, ties, tail blocks, empty outer shapes, noncontiguous
dimension inputs, rejection paths and allocation/input frames. The Python
repair was unchanged in this revision (SHA256
`a0e5722ad5d075978b6019f84060d6f6811f8e3529b8d8f6eea5f47d07f57da9`), so its
previous ten-case Triton 3.7.1 SM80 lowering evidence still applies. No new GPU
execution or independent numerical replay was performed.

Optional proof entries are in `Misc/`: [TensorBridge.lean](Misc/TensorBridge.lean), [NativeCorrect.lean](Misc/NativeCorrect.lean), [ASTCorrect.lean](Misc/ASTCorrect.lean), [ASTHost.lean](Misc/ASTHost.lean), [Audit.lean](Misc/Audit.lean). Their declaration namespaces, theorem statements, proof bodies and existing acceptance/program-pair statuses are preserved. The primary `Correct.lean` and `FPEquiv.lean` entries do not import these optional readers.

The 2026-10-22 source-only cleanup writes all 36 retained canonical typed
configurations as complete ordinary `triton` quotations: six storage choices,
two existing source variants, and three kernels. Those generic historical definitions remain unchanged. The repair now uses
its own raw bodies above. `Proofs/SourcePreservation.lean` proves complete
function equality of the generic historical sources to a frozen proof-only snapshot for each storage and variant,
including the public input/output signatures. Negative-infinity padding,
precision conversions, value/index reduction order, signed-int64 index stores,
masks, dimensions and all source parameters are identical. Compatibility
macros only name these ordinary sources. This historical snapshot equality is separate from the direct execution and
address proofs for the current raw repaired sources. Mathematical contracts,
retained candidate transformations and numerical requirements are unchanged.
