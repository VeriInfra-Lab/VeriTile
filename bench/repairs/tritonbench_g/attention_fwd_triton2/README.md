# attention_fwd_triton2

Full-N base-two normalized attention with D96, query scales per128 and key scales per64.

- [attention_fwd_triton2_repaired.py](Whole001/attention_fwd_triton2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every logical output of complete original 3D grid, any complete grid permutation; five input roles and BF16-tagged output, successful source execution, safety and frame.

## Conditions

- Bases/Domain and output-vs-input separation preserved explicitly.
- Logical96 versus compute128 padding, source TF32 QK, FP16 P/V and PV, FP32 state and BF16 output retained.

Exact FP program pairs and numerical dependencies are recorded in the [attention_fwd_triton2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
