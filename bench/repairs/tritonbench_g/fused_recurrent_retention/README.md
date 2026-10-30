# fused_recurrent_retention

kernel_forward_real and kernel_backward_real: retentionIO implements independent prefix/suffix formulas indexed by actual (part,head,time,channel) and final (head,key,value) coordinates.

- [fused_recurrent_retention_repaired.py](fused_recurrent_retention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Each complete forward or backward NV*NK*BH grid; five masked output channels, optional initial/final-gradient/final-state branches and incoming-gradient strides. Outputs are kernel partials; no private scratch or host-summed dense output is claimed.

## Conditions

Public conditions:

- Shape.Admitted: positive DK/DV and supported BK/BV powers of two
- PlanBinding contains actual valid reduction plans
- incoming and finalGradient stride vectors are explicit
- seed/final flags and all original tagged input values remain
- ProgramIO.Observes keeps explicit node.value observer, physical bindings/safety/frame

Contract and domain correspondence:

Shape admission and plan binding express the existing configuration domain. Independent prefix/suffix formulas use logical DK/DV; displayed partial sums intentionally use source BK/BV with logical tail tests. This distinction is retained rather than relabeling partials as whole host reductions.

Exact FP program pairs and numerical dependencies are recorded in the [fused_recurrent_retention contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
