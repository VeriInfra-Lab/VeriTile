# chunk_gla_fwd

Mathematics.attention explicitly sums logical K causal gated QK terms and scales them. consumerValue explicitly sums logical K gated history terms plus causal attention/value terms. producerResult and consumerResult show the logical output coordinates.

- [chunk_gla_fwd_repaired.py](chunk_gla_fwd_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Two distinct Real stages retained: full rectangular causal attention producer (including both K branches and zero fill), and full standalone consumer. Consumer attention/history remain caller-supplied independent inputs, without claiming an unproved end-to-end composition.

## Conditions

Host.Dimensions and its checked geometry, complete grids, per-call finite IO inputs, source projections, initialized private scratch/output filling and full output frames retained through RealIO AlgorithmFillProgramIO. Contract remains the selected Python wrapper contract with no new tail/alias/block restriction.

Exact FP program pairs and numerical dependencies are recorded in the [chunk_gla_fwd contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
