# swiglu_triton

fun xs ch i => scalarFormula ch (xs.a i) (xs.b i) (xs.dc i)

- [swiglu_triton_repaired.py](swiglu_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One selected contiguous row, parameterized forward/backward stage. Forward writes C; backward reads DC and overwrites A/B, using logical input snapshots and original ordered stores. Main is not a new full-grid result.

## Conditions

- Original cols≤block retained
- PublicIO.formula/output masks select the correct channels per stage
- Public ProgramIO carries input/output windows, safety/success/frame

Exact FP program pairs and numerical dependencies are recorded in the [swiglu_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
