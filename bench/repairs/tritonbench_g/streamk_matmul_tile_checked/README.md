# streamk_matmul

fun xs row col => sum (k : Fin K), xs.a row k * xs.b k col

- [streamk_matmul_tile_checked.py](streamk_matmul_tile_checked.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Selected streamk_matmul_tile_checked revision: full M×N output and full logical K, complete first/reduce/full event stages and all six source helper bodies. Arbitrary initial C/scratch retained, including empty axes/K=0 in the original parameter domain.

## Conditions

- Source.Parameters retains positive blocks, input geometry/contiguity, K8 divisibility, source/configuration/event capacity conditions
- Selected Contract retains existing static tile constraints (BLOCK_K≥16 for nonempty launch and frontend product bounds); no new guard
- Concurrent.EventProgramIO.Implements with EventIO.realArithmetic retains exact event runner and full IO/safety/output/frame obligations

Exact FP program pairs and numerical dependencies are recorded in the [streamk_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
