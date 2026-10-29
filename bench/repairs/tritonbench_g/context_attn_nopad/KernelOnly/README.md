# context_attn_nopad

correct: complete-grid self-inclusive causal attention over keys 0 through query time.

- [context_attn_nopad_repaired.py](../context_attn_nopad_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete logical ordinary-grid output is claimed only with the existing explicit c.completeGrid premise. Tail masks, signed metadata and actual original strided windows remain.

## Conditions

Conditions:

- Host.Shape c
- RealData.Labels r
- complete : c.completeGrid
- RealData.domain c, now displayed with domain-preserving notation

Domain source:

KernelOnly/Contract.lean and Tensor.lean; RealData domain/IO and Proofs/PublicRealResult.lean.

Exact FP program pairs and numerical dependencies are recorded in the [context_attn_nopad contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
