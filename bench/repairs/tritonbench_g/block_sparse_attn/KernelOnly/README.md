# block_sparse_attn

Causal CSR occurrence-multiset attention: full head-feature Q/K dot in each exponential weight; sum weight*V divided by sum weight over all CSR visits and block lanes. Repeated keys remain repeated occurrences, causal keys above the query get zero weight.

- [block_sparse_attn_repaired.py](../block_sparse_attn_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete three-axis selected source grid, all output rows and all 16-wide feature blocks; D=16*blocks. Reversed CSR endpoints mean zero visits, and all admitted occurrences are represented by visits.

## Conditions

Public conditions:

- PublicIO.Geometry p: positive heads and exact output-layout equations
- PublicIO.Admissible: original INT32 endpoint/column ranges, nonnegative first/columns and enough finite visits
- ProgramIO.InBounds backs the actual masked strided windows separately from metadata admission
- Source success, actual safety, tagged output binding and exact frame stay in the judgment

Contract domain source:

Contract.PhysicalKernel.ABI exposes header, CSR guard, actual input/output storage, shape/strides/extents/device/alignment/separation; it contains no execution or desired-output premise. Geometry/Admissible are the public regional mathematical/native domains.

Exact FP program pairs and numerical dependencies are recorded in the [block_sparse_attn contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
