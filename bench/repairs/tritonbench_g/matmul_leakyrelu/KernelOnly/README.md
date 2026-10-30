# matmul_leakyrelu

- Name:

matmul_leakyrelu_real

Statement:

VeriTile.Spec.Real (ReaderIO.view p order ⊨ fun xs r c =>
      let value

- [matmul_leakyrelu_repaired.py](../matmul_leakyrelu_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete original separate dot/add full-grid source in any Schedule p.request, all output coordinates, activation branches, empty axes/K sum, exact exterior frame.

## Conditions

- p.C!=p.A, p.C!=p.B and the complete Schedule remain public; input/input alias scope is unchanged.
- Original ProgramIO typed FP32 input/FP16 output binding and ordinary safety bounds remain in the judgment. The existing Tensor/metadata Contract is untouched.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_leakyrelu contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
