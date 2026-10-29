# triton_mul2

Direct logical pointwise Real arithmetic.

- [triton_mul2_repaired.py](triton_mul2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Two genuine main entries: ordinary destination and in-place load-before-store. Each actual CTA covers pid*B+i<n with exact exterior frame; host grid composition remains outside these established contracts.

## Conditions

- Ordinary/in-place public masked IO judgment supplies active access bounds and input binding; no artificial finiteness restriction in Real
- InputContract retains CUDA contiguous FP32, no negative view, n<2^31, nonempty pointer alignment and supported power-of-two block list
- FP transformation domain is the actual masked multiplicationSite: only active loaded FP32 operands and literal 2 finite

Exact FP program pairs and numerical dependencies are recorded in the [triton_mul2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
