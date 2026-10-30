# rope_embedding

Displayed independent forward/back pair polynomial on every head and rotated column in the active group.

- [rope_embedding_repaired.py](rope_embedding_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One launched row and active four-head group, complete dynamic source loop, both rotated halves, odd tails framed. It is not a whole-grid observation theorem.

## Conditions

Explicit conditions:

- r:Request, back:Bool
- row/group with r.Launched: 0<half, row<rows, group<groups
- ProgramIO includes coherent source-bound inputs, physical windows/masks, access safety, success, all active outputs and exact exterior cell frame.

Contract correspondence:

Original Request domain and both forward/back branches preserved. Spec.pair is the same independent polynomial; coefficients need not have unit norm. Proofs/SourceBinding gives exact local source equality to retained shared FastRopeEmbedding FP32 source.

Exact FP program pairs and numerical dependencies are recorded in the [rope_embedding contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
