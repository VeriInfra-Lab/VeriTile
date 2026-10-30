# chunked_cumsum_fwd

Mathematics.softplus displays thresholded log(1+exp x); prepared displays optional bias, logical validity, lower/upper clamp and padding; cumulative displays inclusive prepared*rate sum.

- [chunked_cumsum_fwd_repaired.py](chunked_cumsum_fwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](KernelOnly/Correct.lean)
- [FPEquiv.lean](KernelOnly/FPEquiv.lean)

## Scope

Complete selected grid and both packed output arrays, including padded zero preparations and continued cumulative sums through padded lanes. All original prefix/FMA topologies remain allowed.

## Conditions

Geometry, Sizes and PrefixPlan remain visible, plus hasBias/reverseBias/useSoftplus and independent runtime lower/UpperBound. The exact selected frozen authorized-minimum-v1 Contract is reviewed and source-checked separately; live Contract is not substituted.

Exact FP program pairs and numerical dependencies are recorded in the [chunked_cumsum_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
