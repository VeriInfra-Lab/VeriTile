# mean_reduction

Optional proof entries are now collected under `Misc/`: [Misc/TensorBridge.lean](Misc/TensorBridge.lean), [Misc/NativeCorrect.lean](Misc/NativeCorrect.lean). Their declaration names, statements, proof bodies and recorded scopes are retained; they are imported separately from the primary entries.

The public proof target is the kernel. `Kernels.lean` gives complete ordinary FP32 quotations for the original
8-by-8 streaming kernel and its existing integer-index commuted variant. The production Python kernel and compression
are unchanged; the second Python body only commutes `program_id * BLOCK_M`.
Every load, FP32 conversion, accumulator update, reduction, division and store
retains its source order. There is no new input copy or dtype family.

`IO.lean` exposes the whole compressed row-major input, every row output,
source-selected FP32 precision, original axis-zero launch metadata and full
arbitrary program order. `Correct.lean` has one public `mean_grid_real`
specification using the shared `UKernelIO.RegionalScheduledImplements` judgment.
It proves the independent finite row mean, successful complete-grid execution,
the complete exterior frame and access safety of the actual source trace.
The kernel theorem includes zero-width reductions and empty grids; it adds no
positivity premise to the original raw theorem.

`FPEquiv.lean` has one public `mean_grid_fp32` specification using shared
`original ≡[[]] indexCommuted` over both complete schedules. The shared
`ProgramIO.Implementation` view retains every actual invocation, observer and
logical input. Its proof uses the original `ProgramIO.Equivalent` judgment.
The independent finite-array
formula retains every FP32 load conversion, source-order addition, selected
per-program reduction primitive and division. Both actual source schedules are
proved safe, successful and equal on all outputs, with the complete frame.
All FP32 words remain admitted. The existing standard `mean_index_equiv`
theorem is retained in `Proofs/KernelNativeResult.lean`; its printed floating
assumption list is empty. The backend family is the primitive model boundary,
not a new numerical rewrite or experimental assumption.

The former conjunction of native formula implementation and schedule
equivalence remains the ordinary `FPEquiv.mean_grid_native_formula` theorem
in `Misc/NativeCorrect.lean`, imported separately as an optional entry. Both judgments keep their
complete grid, arbitrary permutation, successful execution and exterior frame.

`Proofs/SourceSafety.lean` checks the actual streaming body at its actual
predecessor state, using both row and column masks. The candidate's initial
integer multiplication yields the same state before any memory access.
`Proofs/LogicalResult.lean` binds finite logical arrays to the existing
complete-grid execution, and `Proofs/RealIO.lean` and `Proofs/NativeIO.lean`
assemble the shared IO judgments. The exact former raw reader statements
remain ordinary helpers in `KernelRealResult.lean` and `KernelNativeResult.lean`.

`Misc/TensorBridge.lean` retains `mean_dim_kernel_correct`, `mean_kernel_fp32` and
`mean_original_fp32` as ordinary theorems in `TensorBridge`. They connect the
raw results to the checked original Tensor coordinates and the existing
`permute(...).contiguous()` buffer, preserving all storage, copy and view
premises. Empty batches, tails, singleton reductions, empty axis lists and
arbitrary complete schedules retain their previous domain. No finite-value,
non-NaN, zero-add identity or reduction-associativity premise is introduced.
The primary entries do not import this bridge, and proof helpers do not import
the primary entries. Tensor theorem statements are unchanged; their raw helper
references follow the new file locations.

The existing Python contract is retained: strided CUDA FP32 input; omitted or
FP32 output dtype; no unresolved negative view; four-byte-aligned nonempty
input. Noncontiguous and zero-stride inputs remain admitted because the original
compression handles them. Axes are an integer or list of integers, normalized
modulo rank and distinct after normalization; a scalar accepts only an empty
list. Python bool axes follow the source's `isinstance(d, int)` behavior. The
selected extent N is positive; padded M and N are below 2^31, and for M>0,
`(ceil(M/8)*8-1)*N < 2^31`. Empty batches and empty axis lists remain admitted.
`Proofs/Contract.lean` proves successful-domain agreement with these exact
checks, signed indices/masks, and exact active native byte pointers under
ordinary backing spans. No padded allocation is required for masked lanes.
Memory modeling references are centralized in
[`documents/GpuMemoryModel.md`](../../../../documents/GpuMemoryModel.md).

`TensorBinding` binds the original shape/device/pointer observations to the
coordinate address model and valid backing addresses. The pointer condition
is required only for nonempty inputs. Core laws bind the actual compressed buffer. These
are storage/API facts, never assumed output values. Header checks in Python
precede axis normalization, while the typed adapter normalizes first; the
agreement is about admitted inputs, not error priority for multiply-invalid
arguments.

The original unguarded lazy-negative-view counterexample remains: contiguous
compression can retain the negative-view bit, while raw Triton loads read the
unnegated storage. The guarded input contract rejects it. This acceptance is
for the selected checked source, not a claim that the original unrestricted
wrapper is correct.

Historical wrapper theorems are optional in `Proofs/WrapperCorrect.lean`,
`Proofs/WrapperFPEquiv.lean` and `Proofs/Host.lean`. Existing original precision
profiles and proofs remain available. Pure tensor/index/formula definitions
were separated from those host consumers to avoid importing host execution
into the public kernel proofs.

Validation: `python bench/repairs/tritonbench_g/mean_reduction/check_contract.py`
checks unchanged production kernel/compression/wrapper-tail ASTs, the exact
candidate AST change, 15 guard rejections and 60 admissions for each actual
body with exact FP32 word equality. Cases include transpose/sliced/zero-stride
views, nonzero offsets, scalar and empty inputs, boundary tiles and special
values. It executes Python bodies on CPU storage with mocked CUDA; no native
GPU execution is claimed. Targeted Lean build and standard-base axiom audits
are recorded in `/tmp/mean-matrix-reduction-kernel-delivery.json`.

There are no deferred numerical-transformation experiments for this revision.
Independent native replay/GPU execution are not run. Shared runtime grid-product
compatibility remains outside this task.

The 2026-10-20 format pass runs targeted Lean builds and standard-base axiom
audits only, recorded under `.veritile-cache/public-io-format-20261020/` in the
batch04 mean logs/report. The earlier mocked Python checks above are historical
evidence and were not repeated during this format pass.

The source-only preservation theorems in `Proofs/SourcePreservation.lean`
prove equality of both complete source functions to the prior shared typed
family by `rfl`, for every region, extent and block-size argument. Shared
`Whole.source` and `WholeFPValue.source` keep their names and exact values in
`bench/tritonbench_g/mean_reduction/SourceDefinitions.lean`; the old proof
modules import those selectors and retain all execution/proof decomposition.
All existing input domains, precisions, grid schedules and main/optional
statements are unchanged.
