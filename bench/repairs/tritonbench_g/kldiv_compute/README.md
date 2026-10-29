# kldiv_compute

correctness: output[i] = x[i] * Real.log(x[i]/y[i]).

- [kldiv_compute_repaired.py](kldiv_compute_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Real main is the complete n-element grid under positive block size; FP pair is the complete per-program masked window for arbitrary pid, not an asserted whole-grid program pair.

## Conditions

- 0<p.block remains explicit.
- Original IO supports coincident input/output regions and uses initial logical input values; active bounds and whole-grid execution/safety/frame are conclusions.
- Physical Tensor/host contract remains unchanged and optional; no new tensor-layer obligation.

Exact FP program pairs and numerical dependencies are recorded in the [kldiv_compute contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
