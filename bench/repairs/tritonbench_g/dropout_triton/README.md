# dropout_triton

If keep[j]≠0, xs[j]/(1-p); otherwise zero, under total Real division.

- [dropout_triton_repaired.py](dropout_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One masked block at every actual pid in MixedMaskedKernelIO₂; blockSize is source tile width, active index pid*blockSize+j<n. This remains the established per-program theorem, not a newly claimed host/grid theorem.

## Conditions

- Mixed regional IO carries masked FP32 input/output, signed-int32 keep metadata, initial binding/bounds and exterior frame.
- Contract accepts builtin finite Python float p in [0,1], so p=1 is intentionally allowed. The main total-Real theorem is broader.
- Contract.expected explicitly uses total division. Contract.Correct.mathematicalValue is an independent Option-valued conventional formula; MathematicalDomain requires kept active coordinates to have p≠1. At kept p=1 the main Real result is zero by total division, not a conventional defined dropout value.

Exact FP program pairs and numerical dependencies are recorded in the [dropout_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
