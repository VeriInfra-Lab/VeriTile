# f8_conversion_utils

Every active mathematical Real output equals its input.

- [f8_conversion_utils_repaired.py](f8_conversion_utils_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every active coordinate of the selected conversion profile at actual pid; six profiles: e4/e5 decode to half and half/single encode to e4/e5. Source decode retains both stores, masks and profile-selected storage widths.

## Conditions

- MaskedKernelIO₁ declares actual typed source projection, active window bounds, initial binding and output/frame relation.
- Contract input kind/format, device, contiguity, lazy-negation and alignment checks are unchanged. No round-trip or Real-to-FP8 numerical approximation theorem is advertised.

Exact FP program pairs and numerical dependencies are recorded in the [f8_conversion_utils contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
