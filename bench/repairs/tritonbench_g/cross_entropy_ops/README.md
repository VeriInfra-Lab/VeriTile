# cross_entropy_ops

forward_unsplit, forward_split and backward on the original selected source row/block.

- [cross_entropy_ops_repaired.py](cross_entropy_ops_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Unsplit is one covered row; split is one block contribution. Split z-loss is framed by the existing output mask: its displayed zero observation is not a claim that the source writes zero. Backward is one masked row/block invocation.

## Conditions

Conditions:

- Unsplit: row bound, cols>0, cols<=block, split=false, noSmooth, classStart=0, totalClasses=cols, all displayed output separation, RealInputs and valid/ignored label
- Split: row bound, block>0, active start, split=true, noSmooth, output separation and Forward.RealInputs
- Backward: row bound, LSE/logits and LSE/upstream separation, noSmooth and Backward.RealInputs

Domain source:

Interface Forward/Backward RealInputs and IO output masks; checked Contract retains original host guard while the selected theorem remains kernel-only.

Exact FP program pairs and numerical dependencies are recorded in the [cross_entropy_ops contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
