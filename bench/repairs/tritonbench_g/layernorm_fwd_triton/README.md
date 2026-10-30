# layernorm_fwd_triton

Finite column population mean/variance and affine normalization over named tensors.

- [layernorm_fwd_triton_repaired.py](layernorm_fwd_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete sequence/head grid and all three source streaming loops, arbitrary original X/weight/Y strides, every logical column and exterior frame, including empty axes.

## Conditions

- Injectivity of actual yOffset coordinates remains explicit.
- Runtime epsilon ABI is unchanged. Contract requires numeric finite positive epsilon, while the mathematical theorem remains stronger under total Real operations.
- No single-tile cover condition is added to the streaming source.

Exact FP program pairs and numerical dependencies are recorded in the [layernorm_fwd_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
