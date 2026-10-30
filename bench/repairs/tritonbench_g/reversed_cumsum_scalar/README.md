# reversed_cumsum_scalar

For every t:Fin T, sum k:Fin T of if t.val ≤ k.val then xs k else 0.

- [reversed_cumsum_scalar_repaired.py](reversed_cumsum_scalar_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One selected logical row in B*H, all T output positions and the complete reverse source loop including its tail, retaining original strided windows.

## Conditions

Explicit conditions:

- h:Call
- Geometry.OutputLayout h
- c:Config with c∈configurations
- row:Fin(h.B*h.H)
- AlgorithmProgramIO includes coherent physical input binding, output bounds, trace safety, success, all outputs and exact exterior frame.

Contract correspondence:

Contract.mathematical sums the initial input over (range T).filter(t<=k). IOComplete.formula connects the retained suffix formula to the finite-channel target with sum_filter and sum_range. Configurations remain [16,32,32,64,64] by BT with proven positivity.

Exact FP program pairs and numerical dependencies are recorded in the [reversed_cumsum_scalar contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
