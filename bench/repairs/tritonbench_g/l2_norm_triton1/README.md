# L2 norm 1: complete FP32 kernel

The independent Real target is the complete row formula
`y[r,c] = x[r,c] / sqrt(sum_j x[r,j]^2 + eps)`.
`IO.lean` exposes the complete row schedule, FP32 shared source, runtime epsilon
register, typed X/Y channels, dense windows, masks and independent formula.
`Correct.grid_correct` uses the shared `ProgramIO.Implements` judgment: actual
source access safety, successful execution of all rows, every output coordinate,
and the complete exterior typed-cell frame. The reduction cover `N ≤ B` is public.
No raw execution state or per-address result appears in the reader statement.

The previous exact raw row result, arbitrary-heap grid execution certificates,
whole output and exterior-byte frame remain ordinary theorems in
`Misc/RealKernelResults.lean`. `Misc/TensorBridge.lean` retains the original checked
applications using their shared source foundations. The wrapper supplies
assertions; compiler, allocator and host launch internals are not new tasks.

## Source and precision

`Kernels.lean` is the shared readable Triton DSL used by both public entries.
The actual selected Python remains `l2_norm_triton1_repaired.py`. Input, output,
loaded values, arithmetic and reduction are FP32; row/column addressing is
widened to signed64. Runtime epsilon remains an actual FP32 scalar ABI word,
including NaN and infinity. Its parser-only declaration is removed before
execution and the entry state binds the scalar argument. There is no separate
handwritten arithmetic AST or replacement of a literal epsilon site.

`l2_norm_triton1_candidate.py` materializes the existing row-base commutation:
`X += row*N; Y += row*N` becomes `X += N*row; Y += N*row`.
It changes only integer addresses and makes no performance claim. `PairIO.lean` retains the actual original and commuted invocation schedules.
`FPEquiv.source_equivalence` displays `original ≡[[]] commuted`, using the shared
`ProgramIO.Implementation` view and the existing full-grid
`ProgramIO.Equivalent` proof. The prior IO judgment is preserved as ordinary
`FPEquiv.source_io_equivalence` in `Misc/SourceEquivalence.lean`. Both sources
are safe and succeed, all output cells agree, and both exact exterior frames hold
for every scalar backend and input/epsilon value. The stronger source proof yields
identical complete states; the previous public whole-state relation is retained in
`Misc/SourceEquivalence.lean`. Reduction order, casts,
sqrt, division, operand order and masks remain identical. There are **no
numerical transformation assumptions**.

The Real theorem uses total Real sqrt/division, including zero or negative
`squared_norm + eps`. It does not identify those totalized results with native
NaNs/infinities. The native comparison admits every FP32 word. The original
native FMA/register/shuffle/FTZ instruction-graph proofs remain reusable optional
helpers; primitive instruction interpretation is a model/backend boundary,
not an experiment TODO attached to this integer-only equivalence.

## Exact checked domain and storage meaning

The Python checks and `Contract.lean` admit exactly:

- Strided FP32 CUDA tensors of rank at least one.
- Last dimension `1..16384`, with zero leading extents allowed.
- Default-contiguous storage and no unresolved negative-view bit.
- A 16-byte-aligned actual data pointer for nonempty input.

No payload finiteness, positive-epsilon or no-overflow restriction is introduced.
The retained 16-byte check selects the previously reviewed vector/reduction
profiles; it is not a universal FP32 alignment requirement. Nonzero storage
offsets are admitted when their actual pointer satisfies that check.

Optional `Misc/TensorBridge.lean` retains the full checked applications as
`TensorBridge.kernel_correctness` and `TensorBridge.grid_correctness`. It also contains `TensorBridge.python_checks_iff_preconditions`,
which connects the ordered executable checker to
`InputContract`. `Kernel.Buffers` combines ordinary Tensor observations, backing
capacity/separation and equality of dense kernel addresses with the actual
Tensor view keys. Thus the mathematical formula reads the original Tensor
storage, including its offset. `TensorBridge.contract_memory_meaning` reuses the established
contiguous/no-negative/alignment connection. Native active and padded index
bounds are derived from the backing allocations and retained width limit.

The checked wrapper uses two aliasing views and one fresh output allocation;
it does not copy or normalize input values. Empty leading extents launch no
kernel and the grid theorem has an empty output range.

## Original defect and preserved proofs

The original source reused the input row stride for a fresh output. For a
`(4,8)[::2]` input, that stride is 16 while the output row stride is 8, so program
1 writes outside the output. The existing focused checker retains this concrete
counterexample. The current explicit contiguous-input contract makes dense row
addresses agree with the accepted source precision and independent formula.
Original Python is unchanged, SHA-256
`3b190a21ae5ef1c37f8d24e16c1757950215749d28a463f3c1989657fa07a017`.

The prior public host claims remain in `Misc/LegacyCorrect.lean` and
`Misc/LegacyFPEquiv.lean`, with their original scopes. Existing mathematical,
source, physical-memory and native proofs are reused. Compiler/Torch/GPU runtime
internals and extreme-grid compatibility are not new proof tasks here.

## Focused evidence

- `python bench/repairs/tritonbench_g/l2_norm_triton1/check_contract.py`
  checks guards, storage/view behavior and the original stride counterexample.
- `python bench/repairs/tritonbench_g/l2_norm_triton1/check_kernel.py`
  checks both related tasks using unmodified JIT function ASTs under the actual
  Triton CPU interpreter, without replacement reductions or loads/stores.
- `kernel-validation.json` pins current original/candidate hashes and records
  11 pairs / 22 kernel executions for this task. It covers widths 1, 3, 16, 32,
  129, 1025 and 16384, multiple rows, zero/exceptional values and runtime epsilon
  cases. Both versions have identical bytes; input and output guard bytes are
  preserved. Finite controls compare the independent row formula.

These source regression checks do not experimentally validate primitive IEEE
laws. GPU execution and independent proof replay remain separately unperformed.
Numerical deferrals for the current candidate: **none**.

The targeted `Correct` and `FPEquiv` builds and both optional legacy entries
pass. All public axiom checks pass. The current shared structural equivalence
has no floating transformation parameter; the retained historical equivalence
also reports no floating assumptions. This is local validation, not independent replay.

The 2026-10-21 pair-entry review changes only this public equivalence format.
The existing `GridIO.equivalence`, source proofs, actual candidate, runtime
epsilon/input domain, source precision and complete row coverage are reused.

Optional proof applications and reader audits are now under `Misc/`. Their
namespaces, full theorem statements, proof bodies and prior acceptance status
are preserved. `Correct.lean`, `FPEquiv.lean` and their required proof helpers
do not import these optional entries. This directory move adds no correctness
claim or numerical validation.
