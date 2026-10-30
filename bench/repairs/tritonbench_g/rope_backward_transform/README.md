# rope_backward_transform

Displayed adjoint (a*c+b*s, b*c-a*s) on both logical Q/K outputs; every odd tail column returns the original input.

- [rope_backward_transform_repaired.py](rope_backward_transform_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full reverse launchRows schedule, both Q/K in place, independently strided cosine/sine inputs, every result read including odd tails. Writable footprint is only rotated halves.

## Conditions

Explicit conditions:

- Original Request geometry/domain is preserved
- ReadProgramIO distinguishes all result-read cells from writable halves and supplies coherent input bindings, bounds, safe successful execution, all outputs and exact exterior frame.

Contract correspondence:

Contract source_binding is exact to Source.kernel .fp32 .fp32. The independent adjoint formula is not predicated on unit-circle coefficients or inverse rotation; odd dimensions are retained.

Exact FP program pairs and numerical dependencies are recorded in the [rope_backward_transform contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
