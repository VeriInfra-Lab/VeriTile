# int8_dequant_matmul

Current repaired Python implementation: [int8_dequant_matmul.py](../int8_dequant_matmul.py).

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_real: columnScale[col] * (rowScale[row] * sum_k(signed8(../A[row,k])*signed8(../B[k,col])) / 16129) + optional bias[col].

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Source and mathematical adapters](Sources)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.

`Kernels.lean` quotes both complete Python function bodies: the repaired source
(37 top-level statements) and the existing index-reuse candidate (../35).
`jit-correspondence.json` checks every nested statement, cast, branch and mask.
Pointer element types, scalar bindings and launch constexprs are caller metadata.
The explicit signed loop is interpreted by the selected integer-dot model;
bias is loaded as FP32, narrowed to FP16, then added in FP16 before writeback.
The independent Real result, full-source FP comparison and memory safety/frame
proofs retain their existing public input and scheduling conditions.
Current completion requires an accepted independent replay of the exact source
and shared dependencies; source comparison and a build alone do not certify it.
