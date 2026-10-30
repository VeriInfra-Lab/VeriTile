# attention_forward_triton

Full-N scaled base-two normalized attention; head dimension 96, Q scales per128 and K scales per64.

- [attention_forward_triton_repaired.py](attention_forward_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every logical batch/head/query/output coordinate of the full source grid, any complete grid permutation, five finite input roles and FP16-tagged output; safety, success and exact frame.

## Conditions

- Separated and OutputInjective stay explicit in main theorem.
- AttentionIO.RealInputs requires all Real Option payloads isSome; finite channel shape, addresses and scales use logical96 versus padded128 computation. Original grid coverage and all masks retained.

Exact FP program pairs and numerical dependencies are recorded in the [attention_forward_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
