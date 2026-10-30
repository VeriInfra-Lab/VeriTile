# attn_fwd_causal

D128 normalized base-two attention over min(N,row+1) causal keys.

- [attn_fwd_causal_repaired.py](attn_fwd_causal_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete actual two-stage causal source and full grid, every logical output, arbitrary complete grid permutation, source safety/success and exterior frame; empty dimensions/read aliases preserved.

## Conditions

- Output/input separation and complete order retained explicitly.
- Logical/compute feature128, five finite input roles/scales layout, original two-stage branches and per-stage source masks/configuration retained.

Exact FP program pairs and numerical dependencies are recorded in the [attn_fwd_causal contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
