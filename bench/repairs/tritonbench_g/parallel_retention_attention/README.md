# parallel_retention_attention

forward; backward

- [parallel_retention_attention_repaired.py](parallel_retention_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete per-program forward and complete DQ/barrier/DKV backward endpoints at one legal tile. Forward rows use chunk*128, backward rows chunk*64. Scratch shard outputs remain visible; the wrapper later sums shards outside this endpoint.

## Conditions

ProgramDomain s 128/64 tile retains bh/chunk/key/value grid bounds; backward also retains 0<BK and 0<BV. Shape, dtype/device, padding, storage separation and freshness remain in Contract/public IO. The theorem is broader than admitted metadata where its existing mathematical premises suffice.

Exact FP program pairs and numerical dependencies are recorded in the [parallel_retention_attention contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
