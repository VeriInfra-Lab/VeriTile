# llama_ff_triton

- Name:

kernel_correct

Statement:

Spec.Real (ReaderIO.view p ⊨{ReaderIO.byteDomain} fun xs row column =>
      let rho

- [llama_ff_triton_scalar_cast.py](llama_ff_triton_scalar_cast.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected original source grid and both half/int8 storage branches; every logical [rows,columns] output and exact exterior typed-cell frame. Empty outputs and nondivisible inner tails are retained.

## Conditions

- Parameters retains rows, columns, inner, byte and epsilon without new restrictions.
- ArrayProgramIO input binding/read windows and ordinary safety bounds remain in the judgment.
- Interface.byteDomain retains signed-int8 admissibility for all three byte channels, including unused storage-branch ports. Contract.admitted_iff remains the original Python-check correspondence.

Exact FP program pairs and numerical dependencies are recorded in the [llama_ff_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
