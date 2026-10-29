# fused_layernorm_triton

kernel_real: logicalIO implements all-D mean, variance, reciprocal sqrt(variance+1e-5), and affine normalization over named logical tensors.

- [fused_layernorm_triton_repaired.py](fused_layernorm_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected row grid, all logical D columns, both 1024/2048 source chunk sizes, masked chunk tails, FP32 statistics and BF16 output storage. The source loops cover D; no false D<=block restriction is added.

## Conditions

Public conditions:

- 0<D
- Separation X W Bias Y Mean Rstd
- Contract.Configuration.allowed: block=1024 or 2048
- c.tree.Complete for Real mean/variance
- ProgramIO input/physical bounds and frame remain implicit judgment conditions

Contract and domain correspondence:

Contract width positivity and both autotune configurations match the public Real assumptions. Outputs must be fresh as stated by Separation. FP equivalence can retain an arbitrary identical selected tree on both sides; the independent Real formula specifically needs Complete.

Exact FP program pairs and numerical dependencies are recorded in the [fused_layernorm_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
