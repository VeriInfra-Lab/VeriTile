# mul_exponent_compensator

fun xs i => xs i * 2^112

- [mul_exponent_compensator_repaired.py](mul_exponent_compensator_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete arbitrary grid permutation including n=0, tails and in-place src=dst; B>0.

## Conditions

Public conditions:

- 0<B
- order.Perm (List.range (grid n B))
- ProgramIO finite logical binding/bounds and complete exterior frame

Contract relation:

Original contiguous FP32 metadata, supported block list, finite grid/device/alignment checks retained in Contract; formal positive-B source theorem is more general without new host/native/Tensor obligations.

Exact FP program pairs and numerical dependencies are recorded in the [mul_exponent_compensator contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
