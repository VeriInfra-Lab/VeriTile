# max_reduction

masked stage maxima; dimension relation states attainment, upper bounds, least-index ties, and zero value/index at N=0

- [max_reduction_repaired.py](max_reduction_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All three established source results retained: individual stage1 PID blocks, whole stage2 invocation, and every active row in an arbitrary selected dimension program with two typed outputs. No new claim that a per-stage theorem is a composed full-global reduction.

## Conditions

Public conditions:

- stage1/stage2 0<B
- dimension ov != oi, 0<K, ReductionPlan (nextPower N)
- Dimension row mask p*BM+i<M, logical axis count N (not padded nextPower N)
- Shared masked/ProgramIO actual access safety, binding, successful source and exterior frames

Contract relation:

GlobalContract and DimContract retain original FP32 metadata/nonempty logical axis/width/grid checks. Formal empty-axis projection and FP alias/zero-K cases stay precisely at their existing wider model scopes.

Exact FP program pairs and numerical dependencies are recorded in the [max_reduction contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
