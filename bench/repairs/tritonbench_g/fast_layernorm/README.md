# fast_layernorm

Logical-N row mean/variance, affine normalization, saved-stat backward gradient, and all-row DW/DB sums.

- [fast_layernorm_repaired.py](fast_layernorm_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Three existing per-program scopes: a forward row with Y/r/mu outputs, a backward row, and a 128-column affine block looping over every row. N is logical width, B is padded tile. Affine DW/DB output counts wc/bc and optional bias mask remain independent of the input-column N test.

## Conditions

- Forward/backward retain N≤B; backward saved mean and r are independent inputs, not hidden forward-run premises.
- Source-level total Real formulas retain N=0 behavior; checked host Contract separately requires positive bounded width and positive finite epsilon.
- IO exposes input column masks, scalar statistic reads on every affine loop row, independent output masks, bounds/bindings and exact frame; no full-grid claim is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [fast_layernorm contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
