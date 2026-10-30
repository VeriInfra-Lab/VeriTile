# cosine_compute

cosine_kernel_correctness: Real.cos pointwise at each active lane of the original masked CTA.

- [cosine_compute_repaired.py](cosine_compute_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every active lane of an arbitrary original 256-lane CTA with mask pid*256+i<n. The headline is a masked kernel-window result; it does not add a host/full-grid execution claim.

## Conditions

Conditions:

- All n; shared MaskedKernelIO₁ judgment binds the active logical input, output access bounds, finite execution, actual access safety and exterior frame

Domain source:

Contract.mathematical and InputContract; IO.cosine is the single canonical masked IO descriptor.

Exact FP program pairs and numerical dependencies are recorded in the [cosine_compute contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
