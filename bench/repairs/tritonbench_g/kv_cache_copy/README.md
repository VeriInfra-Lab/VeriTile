# kv_cache_copy

LogicalIO.io ⊨{length/block OutputUnique} fun a => ⟨a.key,a.value⟩

- [kv_cache_copy_repaired.py](kv_cache_copy_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One actual seq/head program, including every static-loop split q<dims/width and every width lane. The generic theorem states the exact floor split range. Contract.parameters on checked layouts makes this range cover all logical dims; no full multi-program host grid is newly claimed.

## Conditions

- CopyIO.domain retains OutputUnique on metadata-dependent output addresses across all splits and lanes.
- ProgramIO retains typed input bindings, valid backing windows, source safety, success, both channels and exterior frame.
- Contract InputContract retains shape/layout/device/type, signed metadata bounds, unique destinations, separation and power-of-two geometry. The new display adds no payload or output assumption.

Exact FP program pairs and numerical dependencies are recorded in the [kv_cache_copy contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
