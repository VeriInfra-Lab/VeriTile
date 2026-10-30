# apply_penalty

(if 0 < xs.logits j then xs.logits j / xs.repetition else xs.logits j * xs.repetition) - xs.counts j * xs.frequency - xs.presence

- [apply_penalty_repaired.py](apply_penalty_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One metadata-dependent in-place gather/scatter program, cumsum start/stop active lanes, every active token update, actual source safety/success and exterior frame. FP pair quantifies every program ID.

## Conditions

- MetaScatterMasked2DKernelIO₁ publicly binds metadata, counts, logits, penalty windows, active-ID injectivity, addresses/bounds and frame.
- Contract validates int32 nonnegative IDs/counts/cumsum, capacity window length≤BLOCK, vocabulary bounds, uniqueness and separation; FP rewrite domain is separate from Python admission.

Exact FP program pairs and numerical dependencies are recorded in the [apply_penalty contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
