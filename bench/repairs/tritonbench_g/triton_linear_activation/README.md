# triton_linear_activation

Complete dense K sum plus optional bias, every activation branch, and optional saved preactivation.

- [triton_linear_activation_repaired.py](triton_linear_activation_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete actual invocation with all active block rows/columns, full descending K loop, both FP32 and AMP FP16 storage modes, all bias/activation/saved-output branches. No new host grid obligation.

## Conditions

- 0<p.blockInner; CanonicalStrides p; p.saved≠p.output
- IO retains wrapped input row/column reads, unwrapped active output indices, bias and saved masks; other original alias/stride choices unchanged
- Contract.Domain retains metadata, canonical Tensor binding, optional argument flags, physical separation and SourceBounds, with no output/run premise

Exact FP program pairs and numerical dependencies are recorded in the [triton_linear_activation contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
