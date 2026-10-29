# matmul_persistent_triton

fun (a,b) row col => ∑ k : Fin K, a row k * b k col

- [matmul_persistent_triton_scalar_cast.py](matmul_persistent_triton_scalar_cast.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every persistent PID and assigned row/column tile, the full K loop, masked tails and empty M/N grid; selected FP16 and E4M3 input/output profiles.

## Conditions

Public conditions:

- RawKernel.Layout p b (physical stride/layout data, no execution or desired output)
- 0 < p.problem.SMS
- ProgramIO input binding and spatial InBounds judgment for all finite logical channels

Contract relation:

Contract.layout/strides/inputs/outputBinding/smsPositive is ordinary metadata and backing-memory admission. Correct retains layout/SMS and the public IO binding rather than adding host/Tensor success.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_persistent_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
