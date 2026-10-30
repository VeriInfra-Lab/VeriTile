# cross_entropy2

forward and backward for exactly the original selected row/block programs.

- [cross_entropy2_repaired.py](cross_entropy2_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete forward row/block and one complete backward row/block window. In split mode forward is the local block contribution, without a wrapper reduction or a global unsplit-loss claim.

## Conditions

Conditions:

- Forward: row<rows, block>0, active block start<cols, LSE/logit separation and pairwise separation among LSE/loss/zLoss
- Backward: row<rows; original masked accesses and saved-LSE/upstream inputs retained
- hasSmoothing=false selects zero smoothing in the actual formula

Domain source:

IO forward/backward formula bridges, Proofs/RealIO.lean, and Contract KernelContract/KernelAccess exports.

Exact FP program pairs and numerical dependencies are recorded in the [cross_entropy2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
