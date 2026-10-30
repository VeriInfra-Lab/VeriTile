# triton_conv2d_fwd

Independent full grouped convolution sum over all channel/filter-height/filter-width coordinates.

- [triton_conv2d_fwd_repaired.py](triton_conv2d_fwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Whole original grouped-convolution grid including empty batch, all h/w/c loops, masked tails, FP16/default operand paths and both TF32 flags; no bias/dilation beyond original contract.

## Conditions

- Valid s: positive channels/output channels/spatial/filter/stride/group/output extents, exact floor-divided output shape and positive input/weight strides
- Contract.Domain supplies metadata/shape/stride/device/storage binding, signed index envelope and launch grid checks; main theorem is more general over Valid s
- ProgramIO declares full finite strided inputs/output; actual safety/success/all outputs and exact frame proved

Exact FP program pairs and numerical dependencies are recorded in the [triton_conv2d_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
