# matrix_reduction

Current repaired Python implementation: [matrix_reduction.py](../matrix_reduction.py).

The original block-pointer body and the existing [affine-address candidate](../matrix_reduction.py)
are copied into `Kernels.lean`. Both FP16 and FP32 caller profiles are retained.
The mathematical view is derived from that source through [a checked projection](Projection.lean).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fun xs row _ => sup′ (Fin N) (fun col => xs row col)

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
