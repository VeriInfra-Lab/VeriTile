# flash_decode2_llama

Exponentially weighted split-summary average over ceil(length/Q) summaries at every output feature.

- [flash_decode2_llama_repaired.py](flash_decode2_llama_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete unmasked Llama B*H source grid, every output dimension D, arbitrary order permuting range(B*H), including empty grids.

## Conditions

- Geometry.positiveQ is intrinsic; IO.Lengths requires every launched row to have 0<length≤S*Q, from which count is positive and ≤S.
- IO.outputValues is only a channel adapter. IO has natural length metadata, actual E/M S strides, all dense active windows and exterior frame.
- Selected Contract remains the unmasked checked call domain; no new Positive input guard is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [flash_decode2_llama contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
