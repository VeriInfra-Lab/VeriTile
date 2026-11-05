# pow_scalar_tensor

Current repaired Python implementation: [pow_scalar_tensor.py](../pow_scalar_tensor.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_raw_correctness

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

The complete repaired Python helper and both dispatch branches are copied into
`Kernels.lean`. Scalar types, signed pointer bases/strides and constexpr launch
arguments are supplied outside the quotations. The three exact `_pow` import
fallbacks are checked against Python; helper argument captures retain its one
explicit FP32 cast and opaque power operation.

The comparison [Python source](../pow_scalar_tensor.py) materializes the
retained Lean candidate. Only the two integer `offset0` products commute; its
header records the original module SHA256. Every block-pointer boundary and
load/store cast remains in both quotations. `jit-correspondence.json` checks
both complete callers and their local helper definitions.

Proofs extract the actual branch-local statements. Their complete execution,
signed accesses, output memory and frames connect to the independent power
formula and ordered FP reference. The reference includes the original input
element self-cast; the FP equivalence needs no numerical rewrite assumption.
