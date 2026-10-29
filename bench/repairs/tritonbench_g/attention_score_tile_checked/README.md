# attention_score

N-term signed-window-filtered base-two attention score sum using supplied normalizer, with exact complement/regular window semantics.

- [attention_score_tile_checked.py](attention_score_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full shared source grid and finite output channel, including empty dispatch allocator zero. Logical N sum is independent of B block width; full frame/safety/success retained.

## Conditions

- Whole002.Representation.Domain and live tile bound B*D≤1048576 remain explicit.
- Public initialOutput c only binds input Out=0 for existing empty dispatch (allocator behavior), not successful execution or desired final outputs; nonempty input Out unrestricted.
- Grouped-head mapping, signed offset/size, regular/complement window predicate and supplied M normalizer preserved.

Exact FP program pairs and numerical dependencies are recorded in the [attention_score contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
