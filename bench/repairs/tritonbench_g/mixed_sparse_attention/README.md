# mixed_sparse_attention

independent occurrence-list weighted quotient with base-two exponent and exact coefficient 1.44269504; duplicate columns/blocks retained

- [mixed_sparse_attention_repaired.py](mixed_sparse_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete arbitrary permutation of all grid programs, both original sparse block and column loops, early return, every active query row/channel, duplicate occurrences and empty support.

## Conditions

Public conditions:

- Separated c and Contiguous c
- 0<c.BM and 0<c.BN
- work.Perm (programs (grid c))
- Explicit RealIO.domain c: sequence lengths <=T; input array/read footprints and active output mask remain in ArrayProgramIO
- Original metadata/native accepted storage domain remains in Contract/NativeIO

Contract relation:

KernelContract.Admission/EmptyAdmission and SparseChecks retain all existing checked metadata and storage facts. The logical theorem is more general: it adds no invented sparse-column bound; actual footprints must satisfy public IO bounds.

Exact FP program pairs and numerical dependencies are recorded in the [mixed_sparse_attention contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
