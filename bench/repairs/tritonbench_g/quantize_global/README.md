# quantize_global

kernel_grid_correct

- [quantize_global_repaired.py](quantize_global_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every logical i<n across the complete segmented source schedule and every retained autotune trial; all trials within a segment may overwrite the same output with the same scalar result.

## Conditions

p.Separated; trials(k) nonempty for every active segment; domain=scaleCondition requiring maximum>=0 and abs(x[i])<=maximum. Zero n has no active segment obligation. IO retains integer observations, physical bindings, source safety/success and exterior frame. Contract finite FP32/contiguous CUDA guards remain.

Exact FP program pairs and numerical dependencies are recorded in the [quantize_global contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
