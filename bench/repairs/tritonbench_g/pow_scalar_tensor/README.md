# pow_scalar_tensor

kernel_raw_correctness

- [pow_scalar_tensor_repaired.py](pow_scalar_tensor_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete capped CTA source schedule, both oneTile/grid-stride branches and either existing integer-index spelling, complete signed input/output views and exterior frame.

## Conditions

n>0, input/output Valid for supplied capacities, nonzero output stride, disjoint logical footprints. The preserved mathematical domain requires powerDefined for every exponent: positive base, zero base with nonnegative exponent, or negative base with an integer exponent. This is separate from the all-word FP32 Python input contract.

Exact FP program pairs and numerical dependencies are recorded in the [pow_scalar_tensor contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
