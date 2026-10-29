# reversed_cumsum

fun xs time feature => sum t : Fin h.T, if time <= t then xs t feature else 0

- [reversed_cumsum_repaired.py](reversed_cumsum_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One selected row and one 32-feature block; complete time dimension and feature-tail mask. Main reader honestly states this scope, not a complete row-feature grid.

## Conditions

- input region != output region
- selected c belongs to configurations
- Geometry.Program h is a Row h x FeatureBlock h
- physical IO bounds and input binding

Exact FP program pairs and numerical dependencies are recorded in the [reversed_cumsum contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
