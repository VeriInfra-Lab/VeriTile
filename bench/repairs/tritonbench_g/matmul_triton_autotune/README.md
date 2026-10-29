# matmul_triton_autotune

finite matrix product, then if activation == leaky_relu, max-sign branch with slope 1/100

- [matmul_triton_autotune_view_safe.py](matmul_triton_autotune_view_safe.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete ids permutation of the selected source grid with every original configuration and activation branch. Mathematical Real output is independent of FP16 rounding, which is retained in FPEquiv.

## Conditions

Public conditions:

- ids.Perm (List.range (programs g.request))
- g.C != g.A and g.C != g.B
- Shared ProgramIO logical input/spatial bounds

Contract relation:

MetadataChecks, Selection and Schedule retain original frontend/configuration and program-ID-width observations. The already recorded runtime grid-product compatibility/host application scope is unchanged and is not a new obligation of this review.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_triton_autotune contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
