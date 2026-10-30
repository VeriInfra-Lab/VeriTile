# kldiv_ops

forward: pointwise KL loss when red=0, otherwise an unnormalized Fin cols row sum; only raw-target branch clamps to runtime epsilon. backward: -exp(target) in log-target mode, otherwise -target.

- [kldiv_ops_repaired.py](kldiv_ops_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete forward/backward grids, all rows and chunks, both target modes, every runtime reduction code, runtime epsilon, both empty axes and exterior frame. Host post-normalization is not part of the kernel target.

## Conditions

- 0<block and original Y/O and T/O separation for forward, T/O separation for backward.
- Read-only inputs may alias; epsilon remains a runtime scalar binding.
- No finite-payload guard or target-domain narrowing.

Exact FP program pairs and numerical dependencies are recorded in the [kldiv_ops contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
