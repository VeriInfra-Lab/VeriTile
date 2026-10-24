Optional Tensor/native applications, host results and audit readers are in `Misc/`; canonical `Correct.lean` and `FPEquiv.lean` do not re-export these optional results. Declaration names and proof scopes are preserved.

# L2 norm 2: complete FP32 forward and backward kernels

The independent forward target is
`y[r,c] = x[r,c] / sqrt(sum_j x[r,j]^2 + eps)`.
Writing `s = sum_j x[r,j]^2` and `d = sum_j dy[r,j]*x[r,j]`, the independent
backward target is
`dx[r,c] = dy[r,c]/sqrt(s+eps) - d/(s+eps)/sqrt(s+eps)*x[r,c]`.
The calculus interpretation requires `s+eps > 0`; the source Real theorem uses
Lean's total sqrt/division for arbitrary Real epsilon. This does not equate
native NaNs/infinities with Real results.

`Correct.lean` now has one public IO specification for each genuine kernel:
forward and backward. Both cover the complete logical `[rows, cols]` grid and
use independent mathematical formulas. `IO.lean` lists the exact source
invocations, runtime epsilon register, FP32 source interpretation, dense
input/output windows, masks and empty private-scratch footprint.
`ProgramIO.Implements` proves actual source-access safety, successful execution,
every typed output and the complete exterior cell frame. The identical forward
source and full-grid foundation are reused from `l2_norm_triton1`.

The former raw-row and physical-grid results are retained as ordinary theorems
in `Misc/RealKernelResults.lean`. They continue to cover each actual preceding
physical heap and every exterior byte; the public regional IO judgment does not
replace those physical transport results. Wrapper/allocator/launch execution is
not a new theorem target.

## Canonical source and actual candidate

`Kernels.lean` contains the shared readable DSL: the forward source is reused,
and the backward source retains both masked reductions and the complete final
expression. Input, gradient, output, loads, scalar arithmetic and accumulation
are FP32. All input and runtime epsilon words remain admitted. Row/column
addresses widen to signed64. A parser-only scalar declaration supplies epsilon's
register typing and is removed before execution; the entry ABI binds its value.

The selected Python remains `l2_norm_triton2_repaired.py`.
`l2_norm_triton2_candidate.py` displays the existing integer row-base commutation
for all X/DY/DX/Y pointer bases. It changes `row*N` to `N*row`, leaving every
floating operation, reduction, cast, mask and store unchanged. `FPEquiv.lean`
states `forwardOriginal ≡[[]] forwardCandidate` and
`backwardOriginal ≡[[]] backwardCandidate` for the two complete schedules.
The implementation views retain the actual invocations and observers;
`Proofs/Pair.lean` lifts the existing `ProgramIO.Equivalent` proofs of source
safety, successful runs, equal typed outputs and both exterior frames for
every scalar backend. The original full-state results
remain ordinary theorems in `Misc/SourceEquivalence.lean`. It has **no numerical
transformation assumptions**, including on NaNs, signed zeros, overflow or
nonfinite epsilon. The candidate makes no performance claim.

The retained native schedule helpers keep FP32 casts, FMA/register/shuffle
order, sign-bit negation, `sqrt.approx.ftz.f32` and `div.full.f32`. Their scalar
interpretation and ordinary compiler binding remain model boundaries. They are
not primitive IEEE numerical-experiment TODOs for this address transformation.

## Python checks and actual Tensor memory

Forward uses exactly the `l2_norm_triton1` input contract. Backward requires
strided FP32 CUDA X/DY on the same device, contiguous storage, no negative-view
bits, the same element count, DY rank at least one, DY last dimension
`1..16384`, and 16-byte-aligned nonempty input pointers. X and DY shapes need
not match: row grouping uses the DY last dimension and the output has X's
original shape. Scalar X with a one-element DY and empty leading extents remain
admitted. There are no numerical value guards.

The 16-byte check preserves the previously selected aligned compiler profiles;
it is not the universal scalar FP32 alignment requirement. Nonzero aligned
storage offsets remain valid. No input copy is performed.

`python_checks_iff_preconditions` and `forward_checks_iff_preconditions` connect
the executable checkers to the public contracts. `Kernel.Buffers` binds both
input Tensor views to their actual dense kernel keys, backing allocations,
capacities and output separation. The formulas therefore read the original
Tensor storage rather than unconnected free input arrays. Existing physical
proofs supply exact active/padded native index bounds and exterior-byte frames.

## Preserved work and evidence

The known original row-stride/output-allocation mismatch remains in the focused
source-defect controls. Original Python is unchanged, SHA-256
`386a7d3abb4c3960d6af1ff90cb29c0733c0ac6e9ed51319bd22309568947c12`.
The current explicit input contract avoids hidden layout materialization.
Historical host entries remain optional in `Misc/LegacyCorrect.lean` and
`Misc/LegacyFPEquiv.lean`; their existing theorem scopes are preserved.

Run `python bench/repairs/tritonbench_g/l2_norm_triton2/check_contract.py` for the
retained admission/view/storage checks. Run
`python bench/repairs/tritonbench_g/l2_norm_triton1/check_kernel.py` for both
related tasks' actual Triton CPU-interpreter checks. The script supplies no
replacement reduction, load/store or Boolean semantics.

`kernel-validation.json` pins current original/candidate hashes and records
22 pairs / 44 actual kernel executions for this task. Widths 1, 3, 16, 32, 129,
1025 and 16384, multiple rows, zero/exceptional input and runtime epsilon cases
are covered. Candidate output bytes match exactly; inputs and output guards are
preserved. Finite controls also compare the independent formulas.

GPU execution and independent proof replay are reported separately and have not
been performed in this pass. Extreme-grid compatibility remains deferred outside
the task. Numerical deferrals for these current candidates: **none**.

The targeted public IO, safety/grid foundation, `Correct`, `FPEquiv`, retained
raw helpers and `TensorBridge` builds pass. Strict main-statement surface gates
exclude raw execution/state/memory predicates, and the formula non-circularity
checks and axiom audits pass. The main shared IO equivalences have no numerical
transformation-law parameter; the retained full-state equivalences separately
report empty FP assumption lists. This is local Lean validation, not independent
replay.

`Misc/TensorBridge.lean` retains the forward/backward Python-check correspondence,
contract-to-direct-view binding and complete original application signatures as
ordinary theorems. `Proofs/AccessSafety.lean` follows the actual masks and pointer
expressions for the backward original/candidate sources. `Proofs/GridIO.lean`
composes all row invocations using the existing successful source execution,
readback and frame proofs; it connects declared windows to actual accesses.
The two main entries contain only public IO correctness/equivalence and their
short composition proofs.
