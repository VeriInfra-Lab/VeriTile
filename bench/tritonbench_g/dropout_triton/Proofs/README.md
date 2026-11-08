# dropout_triton

Current repaired Python implementation: [dropout_triton.py](../dropout_triton.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

If keep[j]≠0, xs[j]/(../1-p); otherwise zero, under total Real division.

The two raw quotations retain the complete Python original and existing candidate. Caller metadata selects all three floating storage types and both keep-buffer representations; all twelve original/candidate configurations share those bodies.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
