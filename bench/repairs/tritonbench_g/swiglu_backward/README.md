# swiglu_backward

fun xs ch row col => backwardFormula ch (xs.x row col) (xs.y row col) (xs.dout row col)

- [swiglu_backward_checked.py](swiglu_backward_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full original source grid in every legal permutation, both recomputation branches, all strided matrix windows and tails. Optional OUT store remains selected by R; all three ordered stores retained.

## Conditions

- 0<B
- Ownership c R
- tiles.Perm (grid c B)
- Public ProgramIO preserves active input/output masks, memory binding, safe successful schedule and exterior frame

Exact FP program pairs and numerical dependencies are recorded in the [swiglu_backward contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
