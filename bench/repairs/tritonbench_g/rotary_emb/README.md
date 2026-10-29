# rotary_emb

For each Q/K interleaved pair: even=a*cos_even-b*sin_even, odd=a*sin_odd+b*cos_odd. Four explicit output channels.

- [rotary_emb_repaired.py](rotary_emb_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete source invocation at arbitrary pids, 16 token lanes × 4 head lanes × pairBlock, covering both Q/K head counts with original token/head/rotary masks. Does not relabel a per-invocation theorem as a whole-grid theorem.

## Conditions

- Shape.Valid: positive Q/K heads and width; even rotary prefix bounded by width and both coefficient widths.
- RotaryIO.memory retains separate Q/K counts and coefficient row strides, all four in-place outputs; tail/active masks are in IO.lean.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [rotary_emb contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
