# parallel_attention

forward_grid_real; backward_grid_real

- [parallel_attention_repaired.py](../parallel_attention_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete forward grid and combined DQ/barrier/DKV backward grid. All five channel indices and output masks remain; DQ/DK are value-shard partials. Selected KernelOnly scope excludes host normalization, return-mode policy and later shard sums.

## Conditions

ShapeContract: 0<K<=128, 0<V, BK/BV=selectedTile. key_fits proves K<=BK and there is exactly one key shard; V can span multiple BV shards. Zero B/H/T remain permitted. Shared IO supplies exact logical windows, masks, execution, safety and exterior frame.

Exact FP program pairs and numerical dependencies are recorded in the [parallel_attention contract review](../../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
