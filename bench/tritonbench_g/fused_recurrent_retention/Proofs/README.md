# fused_recurrent_retention

Current repaired Python implementation: [fused_recurrent_retention.py](../fused_recurrent_retention.py).

`Kernels.lean` contains the complete repaired forward/backward bodies and the
existing compare-right candidate: 14 and 23 top-level Python statements in each
file. Runtime and storage metadata stays outside the quotations. The original
`do` identifier, tuple program axes, both backward loops, barriers, masks,
casts and optional state/cotangent paths are retained. All four bodies are
checked independently by `jit-correspondence.json`.

`SourceTypes.lean` stores launch data and independent mathematical reference
models. Execution, output/frame and safety proofs connect those models to the
actual quotations; the reverse `range(../T)` uses the counted-loop constructor and
an existing execution/safety equality with the unit-stride invariant.

[Lean source](../Kernels.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

kernel_forward_real and kernel_backward_real: retentionIO implements independent prefix/suffix formulas indexed by actual (../part,head,time,channel) and final (../head,key,value) coordinates.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
