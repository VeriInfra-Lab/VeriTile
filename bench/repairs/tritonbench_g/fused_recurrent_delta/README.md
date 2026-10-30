# fused_recurrent_delta

fused_recurrent_delta_forward_correctness implements output, residual and finalState; fused_recurrent_delta_backward_correctness implements dq, dk, dv, vectorDBeta, scalarDBeta and dh0 on savedDomain.

- [fused_recurrent_delta_repaired.py](fused_recurrent_delta_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)
- [Whole.lean](Whole.lean)

## Scope

Each complete actual runtime program at explicit (bh,iv), all T steps, all scalar/vector beta and optional state/gradient branches. Forward logical key range is K with BK padding. Backward local value width is BBV; DQ/DK/scalar DBeta are this program's partial sums, not a claimed host reduction across iv. All six backward outputs and both store/reload loops remain.

## Conditions

Public conditions:

- Shape and Flags retain K/V/T and source FBV/BBV/BK geometry
- Runtime scale is explicitly bound in RuntimeIO
- Backward SavedInputs binds only used-time primal q/k/beta, saved residual, incoming gradients, initial state and terminal gradients
- ProgramIO handles actual memory bounds, output masks and frames; no supplied successful-run premise

Contract and domain correspondence:

Contract reuses KernelCheck and finite runtime scale projection; main mathematical Real scale remains arbitrary. SavedInputs is an input/saved-forward-data relation, not a desired output or execution assumption. No new finiteness, positivity, host Tensor or interleaving requirement is introduced.

Exact FP program pairs and numerical dependencies are recorded in the [fused_recurrent_delta contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
