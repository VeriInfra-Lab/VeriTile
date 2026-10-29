# int8_quantization

per_block_int8_real: nearest-away 127*factor*x/blockMaximum, with scale=blockMaximum/127 and the original zero fallback for invalid blocks.

- [int8_quantization_repaired.py](int8_quantization_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full batches×blocks schedule, original query/key branch and active row/channel tail mask, integer code and FP32 scale outputs, including invalid/zero blocks and exterior frame.

## Conditions

- 0<width, 0<blockRows and original factor equality RealSpec.factor query channels.
- FiniteBlock is a branch of the total output formula, not a new input restriction; it preserves the original finite-range bound.
- LogicalBlock filters exactly active lanes; padded width is not claimed to be the logical block extent.

Exact FP program pairs and numerical dependencies are recorded in the [int8_quantization contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
