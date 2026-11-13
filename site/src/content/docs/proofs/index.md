---
title: Proofs & surfaces
description: The theorem surfaces VeriTile exposes, the conventions proofs follow, and the manifest that records what's covered.
---

This section is the **proof side**: how end-user theorems are shaped, the
conventions for writing them, and the bookkeeping that keeps the artifact
honest.

## In this section

- [Theorem surfaces](/VeriTile/proofs/theorem-surfaces/) — the top-level shapes
  (`Spec.Real` and `original ≡[R] candidate`) and the lower-level
  `ComputeCorrect` / `ComputeRefine` interfaces that support them.
- [Correctness surfaces](/VeriTile/proofs/correctness-surfaces/) — surface-by-surface
  guide for picking the right one for a given kernel/spec relationship.
- [Proof conventions](/VeriTile/proofs/proof-conventions/) — naming, structure,
  invariant placement, and the patterns that recur across the bench corpus.
- [Approx-GELU φ strategy](/VeriTile/proofs/approx-gelu-phi-strategy/) — the specific
  numerical-approximation proof strategy used for the GELU / `φ` family.
- [Kernel manifest](/VeriTile/proofs/kernel-manifest/) — what the manifest tracks, the
  schema, and how CI uses it to detect drift.

- [Trust audit](/VeriTile/proofs/trust-audit/) — axiom dependencies, statements, and specification checks.
