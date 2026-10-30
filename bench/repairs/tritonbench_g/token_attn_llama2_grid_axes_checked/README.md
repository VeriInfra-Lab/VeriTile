# token_attn_llama2

fun xs b h t => (sum d, xs.query b h d * xs.keyAt (m.page b t) (h/(heads/kvHeads)) d) / sqrt D

- [token_attn_llama2_grid_axes_checked.py](token_attn_llama2_grid_axes_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Selected grid_axes_checked complete Q/K score grid in arbitrary complete program permutation. Output mask token<len retained; repeated page IDs, original broad strides/read aliases and empty cases retained. No softmax or value aggregation claim.

## Conditions

- Valid c m
- (programs c.grid).Perm work
- Interface.MetadataValues c m remains input domain
- AddressedProgramIO.ImplementsOn retains signed metadata adapter, masks, exact bounds/binding/safety/run/frame
- Selected Contract grid axis bounds and original grouped-head checks retained

Exact FP program pairs and numerical dependencies are recorded in the [token_attn_llama2 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
