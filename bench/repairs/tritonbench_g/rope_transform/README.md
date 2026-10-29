# rope_transform

Displayed forward pair polynomial (a*c-b*s, b*c+a*s) for both logical query/key tensors.

- [rope_transform_repaired.py](rope_transform_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full selected forward row grid, both Q/K in place, padded heads and all selected Request geometries including empty row schedules. Selected Request requires positive even dimensions.

## Conditions

Explicit conditions:

- r:Request preserves original geometry and contiguous read-only C/S contract
- ProgramIO exposes exact in-place Q/K footprints and complete row schedule with source precision, binding/bounds, safe success, every output and exterior frame.

Contract correspondence:

Contract.source_binding is exact. Source supports both directions but selected kernel fixes forward=false; main theorem accurately claims only this selected forward branch. No unit-circle or finite-value restriction introduced.

Exact FP program pairs and numerical dependencies are recorded in the [rope_transform contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
