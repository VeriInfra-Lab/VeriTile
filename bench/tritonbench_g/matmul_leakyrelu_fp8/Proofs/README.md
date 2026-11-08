# matmul_leakyrelu_fp8

Current repaired Python implementation: [matmul_leakyrelu_fp8.py](../matmul_leakyrelu_fp8.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` copies both complete 35-statement Python bodies, including the
leading docstring, nine explicit `int64` conversions, Boolean ceil counts,
dynamic K loop, helper call, masks and FP16 store. Caller bindings and pointer
dtypes stay outside the quotations. The existing
`matmul_leakyrelu_fp8_right_slope.py` candidate is retained unchanged; its helper
exchanges the operands of the FP32 slope multiplication. Both one-statement
helpers are checked separately by `jit-correspondence.json`.

The historical task name contains “fp8”; the selected Python call uses FP32
inputs, FP32 accumulation and FP16 output. The public contracts retain that
scope. The FP comparison uses the existing strict software FP32 multiplication
commutativity certificate; unchanged primitives remain explicit model inputs.
It does not claim hardware conformance or a new numerical experiment.

This source migration remains pending until independent proof replay accepts
the exact current source and import closure. Body matching and local builds
alone do not complete the task.
