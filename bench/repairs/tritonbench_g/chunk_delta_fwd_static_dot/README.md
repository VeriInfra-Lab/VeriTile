# chunk_delta_fwd

Correct.Mathematics.corrected = value minus the finite logical-K delta/state contraction; Mathematics.state is the independent seed/rounded-window update recurrence. formula routes history, corrected values and optional final state.

- [chunk_delta_fwd_static_dot.py](chunk_delta_fwd_static_dot.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected static-dot FP32 grid, every CTA permutation. Time strides are BT, rounded update width is ((BT+BC-1)/BC)*BC: overlap and padded/partial chunks are retained, rather than replaced by a disjoint ideal recurrence.

## Conditions

Parameters, Geometry.ShapeValid p and schedule.Perm (grid p) remain visible. Interface IO retains original weight strides, bounds, optional seed/final masks, typed channels, safety/success and exact exterior frame. Selected Contract adds only StaticDot at nonempty B*H*V, equivalent to K>=9 and BT>=16, in addition to base Admitted.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_delta_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
