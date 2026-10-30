# relu_strided_buffer

relu_strided_buffer_kernel_correctness

- [relu_strided_buffer_repaired.py](relu_strided_buffer_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original capped grid and both oneTile/grid-stride branches, complete logical signed input/output views and exterior typed-memory frame.

## Conditions

n>0; input/output Valid for supplied capacities; nonzero output stride; SignedGrid.FootprintsDisjoint. These match checked positive extent and ordinary signed views, including negative input/output strides where valid; no restriction added.

Exact FP program pairs and numerical dependencies are recorded in the [relu_strided_buffer contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
