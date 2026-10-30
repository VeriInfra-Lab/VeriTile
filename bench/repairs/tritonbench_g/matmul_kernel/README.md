# matmul_kernel

- Name:

correct

Statement:

VeriTile.Spec.Real (ReaderIO.view p ⊨ fun xs r c => ∑ t : Fin p.K, xs.A r t * xs.B t c)

- [matmul_kernel_repaired.py](matmul_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete two-dimensional grid and full M*N result, K-tail masks/modulo loads, K=0 and empty outer axes, exact exterior frame.

## Conditions

- p.Valid is unchanged: original block, matrix view, writable output geometry and source index domain.
- ProgramIO logical M*K/K*N input channels, original typed binding and ordinary access bounds remain; no desired-result or callback premise is public.
- Contract.admitted_domain retains the original checked guard-to-domain correspondence.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
