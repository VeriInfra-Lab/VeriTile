# softmax_optimize

For each logical output j: exp(xs[j])/sum_{i:Fin N} exp(xs[i]).

- [softmax_optimize_repaired.py](softmax_optimize_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected row, all N entries across the original four loops (full and final tail in each pass). Config.tile N may be smaller than N; the sum and IO arities use N. Main is per row at arbitrary source pid, not a new whole-matrix theorem.

## Conditions

- N>0 and Y≠X unchanged.
- Config.tile=min(4096,2^N.log2); tile_positive/tile_le_width and previous_eq_full_tiles/tail_covers show full/tail coverage without N≤tile.
- UKernelIO regional judgement retains all launch coordinates, physical input/output windows and raw exterior frame.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [softmax_optimize contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
