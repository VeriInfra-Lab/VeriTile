# rmsnorm_triton

(xs batch row k * sqrt ((sum j:Fin width, xs batch row j * xs batch row j)/width + eps)^(-1)) * weights k.

- [rmsnorm_triton_repaired.py](rmsnorm_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete 2D batch/row grid, arbitrary original tensor strides, both complete source loops with blockSize=1024, unpadded logical width sum.

## Conditions

Explicit conditions:

- g:KernelGrid.Geometry
- Separation g.parameters retained in main theorem
- OutputUnique g.parameters
- work:Schedule g.parameters supplies the complete schedule
- RegionalScheduledImplements explicitly uses RmsIO.schedule g work, retaining coherent typed IO, access safety, success, every row output and raw-cell exterior frame.

Contract correspondence:

Contract.mathematicalResult uses the same unpadded RMS expression on original input and weight tensors. IO.rms/target preserve logical width independently of source block size. Existing Schedule and OutputUnique retain coverage and singleton zero-stride behavior.

Exact FP program pairs and numerical dependencies are recorded in the [rmsnorm_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
