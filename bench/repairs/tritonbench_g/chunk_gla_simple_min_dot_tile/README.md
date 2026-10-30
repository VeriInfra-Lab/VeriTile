# chunk_gla_simple

Mathematics.qhSum/qkSum display block/lane contractions; score displays the causal gated QK weight; expected displays gated history plus causal weighted V; result gives logical full-output coordinates.

- [chunk_gla_simple_min_dot_tile.py](chunk_gla_simple_min_dot_tile.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original min-dot-tile shape, all logical heads*T*V outputs and original 3D schedule. Padded BK lanes retain canonical masks; logical K is not replaced by a single tile.

## Conditions

0<c.BT retained. RegionalScheduledImplements and PublicIO.realSchedule retain the accurate regional scheduled execution, access safety, typed output and complete exterior frame judgment. Specialized spelling is intentionally preserved.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_gla_simple contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
