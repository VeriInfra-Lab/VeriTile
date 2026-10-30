# layer_norm_ops

Independent LN/RMS/residual/recompute/bias/parameter-gradient formulas with explicit named multi-output observations.

- [layer_norm_ops_repaired.py](layer_norm_ops_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

The complete original forward program and bounded backward row-block source loop, with every active branch/output, including empty-loop weight reads and all recompute/residual/RMS flags.

## Conditions

- KernelExecution.ForwardAdmissible or KernelReadback.Admissible and columns≤block remain explicit.
- All region/stride alias conditions, optional masks and saved-statistic input domains are retained.
- The logical sums range over columns; DW/DB range over the actual row interval. No desired-output premise is added.

Exact FP program pairs and numerical dependencies are recorded in the [layer_norm_ops contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
