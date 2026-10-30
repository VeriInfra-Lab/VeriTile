# int8_matmul_kernel

kernel_low32: signed low32 of the four packed-field dense integer sum, using the original defined FP32-to-byte cast.

- [int8_matmul_kernel_modular_checked.py](int8_matmul_kernel_modular_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full M×N production grid for each of the fifteen retained legal configurations, including overflow, K=0 and empty output axes.

## Conditions

- DefinedCast on every FP32 operand, packed uint8 logical values below 256, and a cast implementation agreeing with referenceCast on that exact domain.
- Legal configuration, original OutputLayout and output separation from input/weight; original input strides and input/weight aliasing remain.
- ProgramIO.ImplementsOn is displayed as ⊨{PublicIO.Domain layout s}; bounds, actual execution, output cells and frame remain in the judgment.

Exact FP program pairs and numerical dependencies are recorded in the [int8_matmul_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
