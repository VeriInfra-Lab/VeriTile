# kv_cache_filling

Full-head finite minimum/maximum over dk/dv, scale 1 on constant rows and otherwise (hi-lo)/levels; byte floor((x-lo)/scale+1/2); INT8 levels=255, INT4 levels=15 with low+16*high wrapping packing; copy and both metadata lanes.

- [kv_cache_filling_repaired.py](kv_cache_filling_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](KernelOnly/Correct.lean)
- [FPEquiv.lean](KernelOnly/FPEquiv.lean)

## Scope

Actual complete copy/INT8/INT4 DSL programs, valid source accesses, successful execution, every cache/metadata output and exact exterior frame. The family is not a composed host execution.

## Conditions

- Original Request/Domain metadata and shape constraints.
- Exactly the decoded source/destination input integers; independent entry states for each scheduled job.
- All policies, both versions, all pids; positive logical widths and next-power-of-two coverage follow from Domain/library lemmas.

Exact FP program pairs and numerical dependencies are recorded in the [kv_cache_filling contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
