# fifth_order_sph_harmonics

Eleven exact decimal-coefficient degree-five polynomials and their mathematical Jacobian-transpose application.

- [fifth_order_sph_harmonics_repaired.py](fifth_order_sph_harmonics_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete forward/backward grids. Forward computes all eleven output channels at columnOffset; backward sums all eleven upstream channels into all three coordinate gradients. Source width fit columnOffset+11≤outputWidth remains explicit.

## Conditions

- Real backward retains prepared p: actual active initial coordinate-gradient inputs are zero. This is independent initial data, not a source run/result premise.
- Forward/backward IO retains actual row count, tile count, output strides, all channel masks and bounds/bindings.
- Contract preserves positive valid block metadata and destination coverage/separation; no extra numerical input restriction is added.

Exact FP program pairs and numerical dependencies are recorded in the [fifth_order_sph_harmonics contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
