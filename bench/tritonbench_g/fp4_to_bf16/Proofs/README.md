# fp4_to_bf16

Current repaired Python implementation: [fp4_to_bf16.py](../fp4_to_bf16.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

fp4_to_bf16_real: logicalRealIO implements, on LogicalByteDomain, xs ↦ (j ↦ some (nibbleReal (logicalNibble xs j))).

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
