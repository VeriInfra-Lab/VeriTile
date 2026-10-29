# decay_cumsum

decay_cumsum_kernel_real for each forward, prepare and backward main/tail Program.

- [decay_cumsum_repaired.py](decay_cumsum_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Every lane of each selected complete main/tail invocation for all three kinds. IO covers p.span times h.BK in the existing padded packed feature window; this is not a new full-wrapper or unpadded global Tensor theorem.

## Conditions

Conditions:

- Kind is universally quantified, preserving forward/prepare/backward
- hp : p∈phases h retains both existing main and tail choices
- Program h p explicitly carries feature/chunk/row bounds
- Call carries positive time and feature tiles; DecayIO retains actual in-place dqo/dko and raw-cell bounds/frame

Domain source:

Kernels Call/phases, Proofs/Geometry.Program and IO.lean canonical channels/read/write/schedule.

Exact FP program pairs and numerical dependencies are recorded in the [decay_cumsum contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
