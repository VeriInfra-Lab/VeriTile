# cross_entropy1

forward_unsplit_kernel, forward_split_kernel and backward_kernel on the same selected row/block source programs.

- [cross_entropy1_repaired.py](cross_entropy1_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Three actual kernel results: one unsplit row, one split block contribution, and one backward row/block window. Split is not a host aggregation; backward consumes independently supplied saved LSE and upstream.

## Conditions

Conditions:

- Unsplit: row<rows, LSE/logit and LSE/loss separation, cols>0, cols<=block, split=false, noSmooth, classStart=0, totalClasses=cols and unsplitDomain valid/ignored label
- Split: row<rows, same separation, block>0, active block start<cols, split=true and noSmooth
- Backward: noSmooth; original ignored-label upstream mask, active-column store mask and in-place IO retained

Domain source:

Proofs/RealIO.lean unsplitDomain; IO.forwardFormula/backwardFormula; Contract exports KernelContract and KernelAccess.

Exact FP program pairs and numerical dependencies are recorded in the [cross_entropy1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
