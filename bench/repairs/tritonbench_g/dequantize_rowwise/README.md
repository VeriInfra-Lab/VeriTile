# dequantize_rowwise

dequantize_rowwise_correct: scale * signed weight / 127 at every original active output.

- [dequantize_rowwise_repaired.py](dequantize_rowwise_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every active feature of an arbitrary selected row-width invocation, with the exact j<cols mask and one unmasked scale read, including empty rows. No full host/grid claim is added.

## Conditions

Conditions:

- Parameters retains its existing cols<=width coverage and row geometry
- Shared MixedMaskedKernelIO₂ judgment binds active integer weights, one scale lane, actual access bounds/safety, output and exterior frame

Domain source:

Kernels.Parameters and canonical IO.dequantizeIO; Contract retains INT8 weights, FP32 scales and original metadata checks.

Exact FP program pairs and numerical dependencies are recorded in the [dequantize_rowwise contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
