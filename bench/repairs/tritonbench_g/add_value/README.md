# add_value

xs i + 10

- [add_value_repaired.py](add_value_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete selected add10 program at arbitrary program ID, B=1024, masked tail, successful execution, safety and exterior frame; no full-grid Real claim.

## Conditions

- Public masked IO finite input/output bindings and bounds; wrapper contiguous CUDA FP32, alignment, numel<2^31 and empty behavior retained.

Exact FP program pairs and numerical dependencies are recorded in the [add_value contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
