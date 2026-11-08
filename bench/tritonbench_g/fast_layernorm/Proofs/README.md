# fast_layernorm

Current repaired Python implementation: [fast_layernorm.py](../fast_layernorm.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Logical-N row mean/variance, affine normalization, saved-stat backward gradient, and all-row DW/DB sums.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The three original `@triton.jit` bodies in `fast_layernorm.py` are quoted in
`Kernels.lean` statement for statement, including pointer updates, every `int64`
cast, the affine `while`, its scalar `tl.full` counter and final increment.
`fast_layernorm_candidates.py` records the existing candidate transformations:
commute only the X row-index multiplication in forward/backward; remove only the
inactive-column select from the affine address while retaining the load masks.
`jit-correspondence.json` checks all six complete bodies. Caller pointer dtypes,
base addresses, scalar geometry and constexpr bindings remain outside the quotes.

The proof uses the actual FP32 reads followed by explicit source casts, integer
count conversion, source reduction order and implicit stores. The affine
invariant proves every signed iteration and accumulator, successful termination
of the source while, both output masks, all memory accesses and the exterior
frame. The Real formulas and complete FP source-pair equivalences keep their
established domains; source comparison and successful compilation alone do not
certify completion. Exact independent proof replay is recorded separately in
`validation/body-migration-progress.json`.
