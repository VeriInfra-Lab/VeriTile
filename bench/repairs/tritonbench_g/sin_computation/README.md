# sin_computation

Real.sin(xs[i]).

- [sin_computation_repaired.py](sin_computation_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected per-program tile of 4 lanes with active mask pid*4+lane<n. The result includes source trace safety, active values and masked/exterior frame; does not relabel kernel-only proof as host/full-grid execution.

## Conditions

- Original logical n and masked IO bounds/bindings; all pids quantified by the shared MaskedKernelIO₁ judgment.
- Tensor Contract retains FP32 contiguous CUDA layout, extent/alignment and original alias policy; no host/native/Tensor obligation added.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [sin_computation contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
