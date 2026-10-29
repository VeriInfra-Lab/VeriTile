# cache_transform_triton

Every output byte j is payload[rows[j/rowBytes]*rowBytes + j%rowBytes], with signed row metadata interpreted under its public nonnegative/range domain.

- [cache_transform_triton_repaired.py](cache_transform_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All p.count output bytes of the actual START=0, BLOCK=256 source grid, including the final partial block and empty launches. This is byte gathering, not a claim about FP32 rounded values.

## Conditions

Public conditions:

- p.start=0
- p.block=256
- blocks p.count≤gridLimit
- CacheIO.Domain: on live calls count=n*rowBytes, positive rowBytes, output separation, capacity≤i64Max and every actual metadata row within capacity
- Exact-cell input/tag binding and ProgramIO access bounds retain the unconditional metadata-row load, masked payload load/store, safety, success, outputs and frame

Contract domain source:

Contract.Request checks actual INT32 length metadata, FP32 contiguous cache shapes, devices and length/native bounds. The regional theorem uses prepared signed-int64 rows and uint8 payloads; the unchanged exact-cell input mode and domain maintain this distinction.

Exact FP program pairs and numerical dependencies are recorded in the [cache_transform_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
