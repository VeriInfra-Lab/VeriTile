# triton_argmax

Least maximizing logical index, independently defining maximality and first-index ties, for both global and dimension branches.

- [triton_argmax_nan_first.py](triton_argmax_nan_first.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Real global: all logical c.size values, complete first-grid plus second stage. Real dimension: one actual dimension CTA with full N reduction and active BM rows at ((pid+base)/K)*BM+j, inner coordinate (pid+base)%K. FP global/dimension entries retain their complete grids and empty-outer cases.

## Conditions

- Global: X≠V, X≠cast I, V≠cast I; Math.GlobalCoverage has positive logical size/block/count, complete trees, full size coverage, each first CTA active, and mid scratch fit
- Dimension: 0<BN, 0<N, 0<K; original BM/base/pid/index64 and original active row mask
- FP global retains Request.Casts/Kind and BoundSource.Binding for all thirteen requested scratch formats, Native.GlobalPrograms, positive block/midBlock, active CTAs and NativeIO.initialized gather tail
- FP dimension retains nonempty reduction extent, original DimConfig, index-width branch and all raw words

Exact FP program pairs and numerical dependencies are recorded in the [triton_argmax contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
