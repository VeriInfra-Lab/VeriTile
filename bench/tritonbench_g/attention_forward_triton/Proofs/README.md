# attention_forward_triton

Current repaired Python implementation: [attention_forward_triton.py](../attention_forward_triton.py).

[Lean source](Support/KernelEntryAdapters.lean) · [input contract](../Contract.lean) · [Real correctness](../RealCorrect.lean) · [FP equivalence](../FPEquiv.lean).

Full-N scaled base-two normalized attention; head dimension 96, Q scales per128 and K scales per64.

The public entries state the input conditions, execution scope and numerical assumptions. Required proof and lowering helpers support this selected implementation; Git retains prior implementations.

[Proof helpers](.)

[Selected-task and source-correspondence records](../../../../validation) track validation and pending work separately.
