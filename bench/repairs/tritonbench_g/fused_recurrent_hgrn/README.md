# fused_recurrent_hgrn

kernel_correct: full forward/backward logicalIO implementation conjoined with the exact directional derivative of the adjacent scalar loss objective, using the same dx/dg.

- [fused_recurrent_hgrn_repaired.py](fused_recurrent_hgrn_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete forward grid followed by complete backward grid, every BH/T/D lane, independent BDF/BDB, T=0, all U/S/V flags and optional final-state output. Logical D tail coverage follows the existing positive-width grid theorem.

## Conditions

Public conditions:

- 0<BDF and 0<BDB
- U initial-state flag, S final-output flag and V terminal-gradient flag stay independent kernel parameters
- ProgramIO masks bind h0 iff U, dht iff V and ht output iff S
- Physical memory bindings/bounds and frame remain in the judgment

Contract and domain correspondence:

Checked host contracts supply admissible shapes, FP32 allocation metadata, legal tile sizes and their allocation separation. The main kernel theorem already supports the broader independent flag/width parameters; it does not claim a host call with absent dht has satisfied V=true input binding.

Exact FP program pairs and numerical dependencies are recorded in the [fused_recurrent_hgrn contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
