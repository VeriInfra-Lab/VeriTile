# add_example

xs i + ys i

- [add_example_repaired.py](add_example_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete selected kernel program, arbitrary program ID, active masked lanes and tails, successful regional execution, access safety and exact exterior frame; no full-grid Real claim.

## Conditions

- Public regional IO input/output windows and bounds; B is wrapper blockSize=1024.
- Read-only input aliases admitted; Contract has contiguous CUDA FP32, alignment, numel<2^31 and empty launch behavior.

Exact FP program pairs and numerical dependencies are recorded in the [add_example contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
