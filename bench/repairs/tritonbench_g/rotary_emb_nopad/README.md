# rotary_emb_nopad

(a*c-b*s, a*s+b*c), explicitly for Q-first/Q-second/K-first/K-second; cached K values are written into the metadata-selected cache.

- [rotary_emb_nopad_repaired.py](rotary_emb_nopad_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Two genuine main results: complete uncached per-tile invocation (4×1×blockPairs) with GQA-leader K branch, and complete cached per-tile invocation with early head return and all Q/cache outputs. Original per-tile scope is retained.

## Conditions

- Uncached: ∀ key, 0 < g.request.heads key. Cached: pid1 < g.tokens.
- RotationIO.MetadataDomain: 0≤contextLength-1 and 0≤page; signed metadata values are logical inputs.
- uMask/cMask and coefficient masks preserve tail, GQA/early-return behavior; IO carries exact K/exterior frame.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [rotary_emb_nopad contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
