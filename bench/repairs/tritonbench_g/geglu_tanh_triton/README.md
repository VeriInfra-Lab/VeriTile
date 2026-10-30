# geglu_tanh_triton

geglu_kernel_correctness: logicalIO implements geluValues ch a[i] b[i] (incoming[i] in backward, otherwise zero).

- [geglu_tanh_triton_repaired.py](geglu_tanh_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every count element over any complete legal 128-lane grid order; original/candidate and forward/backward branches. Backward reads entry A/B/DC before in-place A/B stores; output masks select C versus DA/DB.

## Conditions

Public conditions:

- order.Perm (List.range (Grid.grid count))
- ProgramIO finite logical channel windows, per-mode output masks, actual memory bounds and frames
- Arbitrary mathematical Real inputs; finite refers to channel index sets, not a numeric finiteness restriction

Contract and domain correspondence:

Contract preserves FP32 storage, matching input counts, original shapes/device, positive cols, native indexing bounds and allocation separation. The independent exact-decimal GELU approximation is the same one used by the source and canonical Math derivative proof.

Exact FP program pairs and numerical dependencies are recorded in the [geglu_tanh_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
