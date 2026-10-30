# spinning_lock_reduction

Named partials/output/locks stage formula adjacent to the specification

- [spinning_lock_reduction_phased.py](spinning_lock_reduction_phased.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete actual concurrent producer and consumer stages, original CAS retries/barriers/interleavings, all selected threads complete. Output formula includes untouched cells, holes and overlapping output strides; no sequential replacement or whole-matrix tiling claim.

## Conditions

- Wide.Buffers h: metadata/capacity/separation/fixed-width bounds
- IO.domain 0: Published (lock=1 implies corresponding partial lanes zero) and consumer stage implies Ready
- Concurrent.EventProgramIO.ImplementsOn retains explicit real backend and source decode, actual run, access safety, input/output bindings and frame

Exact FP program pairs and numerical dependencies are recorded in the [spinning_lock_reduction contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
