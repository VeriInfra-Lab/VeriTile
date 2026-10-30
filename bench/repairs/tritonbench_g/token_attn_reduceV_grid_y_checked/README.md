# token_attn_reduceV

fun xs b h d => sum n in range (counts b), xs.probAt b h n * xs.valueAt b h n d

- [token_attn_reduceV_grid_y_checked.py](token_attn_reduceV_grid_y_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Selected grid_y_checked complete B×H grid in every permitted permutation, all output dimensions, gathered page/value rows and original tails. Zero-block/empty behavior remains in FP; Real theorem retains its preexisting positive block requirement.

## Conditions

- (Grid.fullGrid batches heads).Perm work
- 0<p.block for Real result
- Grid.Separate p and Grid.Unique p work
- Interface.LengthsMatch matches counts to actual metadata
- Original derived finite capacity, strides, read aliases and Contract gridY guard retained; no extra count/maximum guard

Exact FP program pairs and numerical dependencies are recorded in the [token_attn_reduceV contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
