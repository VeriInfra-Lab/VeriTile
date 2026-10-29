# chunk_linear_attn

Mathematics.state/output/finalState show independent causal attention sums; adjoint/gradients show complete VJP sums; result routes all six mathematical output channels.

- [chunk_linear_attn_repaired.py](chunk_linear_attn_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All four complete grids plus unchanged finite FP32 sum, optional endpoints, legal arbitrary per-stage permutations. Full independent IsVJP conjunct retained; no reduction to forward-only or one-stage behavior.

## Conditions

Config, Host.Valid dimensions and KernelSchedule.Legal remain visible. ReductionProgramIO retains six inputs/six outputs, three private buffers, optional masks, safety/success and exterior frame. Contract preserves original wrapper flags, Tensor API and pointer guards.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_linear_attn contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
