# swiglu_fwd

fun xs i => xs.x i * xs.y i / (1 + exp(-xs.x i))

- [swiglu_fwd_checked.py](swiglu_fwd_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One actual (row,tile) invocation with two program axes, arbitrary selected strides, active columns tile*B+j<cols and tail masks. Main does not claim full-grid coverage.

## Conditions

- Original Layout and B,row,tile parameters unchanged
- ProgramIO public bound/input/output obligations preserve source access safety, active output window, successful execution and frame

Exact FP program pairs and numerical dependencies are recorded in the [swiglu_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
