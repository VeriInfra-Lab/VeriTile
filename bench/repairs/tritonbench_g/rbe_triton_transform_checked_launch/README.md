# rbe_triton_transform

kernel_correct

- [rbe_triton_transform_checked_launch.py](rbe_triton_transform_checked_launch.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Selected checked-launch revision, complete two-axis schedule qs.Perm(Runtime.programs r), every contiguous logical output cell, odd tails and empty shapes, signed positions, both barriers.

## Conditions

X!=Y and exact complete schedule retained. Contract adds only nonempty batch<=2^31-1 and launchGrid<=65535; domain_exact/empty_domain_unchanged explicitly preserve original empty-shape admission. Physical capacities/binding and frame stay in shared IO.

Exact FP program pairs and numerical dependencies are recorded in the [rbe_triton_transform contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
