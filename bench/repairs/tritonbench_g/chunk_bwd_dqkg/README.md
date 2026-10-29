# chunk_bwd_dqkg

Independent dQ/dK contractions over all logical V with causal 64-token score terms and exponential gates; dG is the per-key-tile BK contraction with the terminal chunk correction. All three formulas are displayed together.

- [chunk_bwd_dqkg_repaired.py](chunk_bwd_dqkg_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Source.lean](Source.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected original source grid and all three logical output channels under every complete CTA permutation, including empty axes. dG remains unreduced per key tile; this theorem does not claim the separate downstream reverse cumulative sum.

## Conditions

Public conditions:

- order.Perm (allCTAs c)
- Config retains source shapes, 64-token time blocks, logical dimensions and tile/configuration constraints
- PreparedAlgorithmProgramIO binds the seven independent input arrays, all actual access capacities, successful full source schedule, all three outputs and exact exterior frame

Contract domain source:

Contract aliases KernelContract.check/check_iff and Scale; these retain the actual checked kernel ABI. Source.lean and Kernels.lean keep FP32 selected storage, zero-padding block masks, history chunk indexing and the original arithmetic order.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_bwd_dqkg contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
