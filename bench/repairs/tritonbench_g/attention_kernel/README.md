# attention_kernel

Full-N base-two softmax weighted value sum with source-exact 1.44269504 coefficient and both bias windows.

- [attention_kernel_repaired.py](attention_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original grid with every logical output and arbitrary complete schedule order; actual safety/success/frame and separate machine Closed conclusion. Both original FP16/BF16 storage branches remain represented.

## Conditions

- Domain, hbm>0, Separated and complete grid permutation retained explicitly; original strides, biasLast/blockN indices, output mapping and source configuration retained.

Exact FP program pairs and numerical dependencies are recorded in the [attention_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
