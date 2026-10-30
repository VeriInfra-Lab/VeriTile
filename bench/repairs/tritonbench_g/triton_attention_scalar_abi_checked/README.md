# triton_attention

Complete causal attention and Q/K/V VJP using independent full-sequence probability and score-cotangent sums.

- [triton_attention_repaired.py](triton_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquivBound.lean](FPEquivBound.lean)

## Scope

Complete selected forward/preprocess/backward materialized schedules and all Out/DQ/DK/DV cells, saved scratch and exact exterior frame; initialized in-place DQ retained.

## Conditions

- Shape carries positive geometry and original tile geometry; all three Grid.FullSchedule premises remain public
- MaterializedProgramIO.ImplementsOnCells (PublicIO.ZeroDQ 0 c): actual input cells and zero-initialized DQ, bounds, materialized source execution/safety/frame
- Contract preserves raw scalar positivity/finiteness and unsigned64 integer ABI bound; FP pair keeps ScalarABI class, owners, instruction interpretation and literal binding

Exact FP program pairs and numerical dependencies are recorded in the [triton_attention contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
