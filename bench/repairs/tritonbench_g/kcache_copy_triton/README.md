# kcache_copy_triton

kcache_copy_triton_real: output[lane] = payload[lane].

- [kcache_copy_triton_repaired.py](kcache_copy_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One full (token,head,split) tile with all lanes active; signed length/table metadata, dynamic route, original input aliases and snapshot scatter semantics are retained. No full-host-operation claim.

## Conditions

- CopyIO.OutputInjective p ids remains the visible logical input domain for mathematical readback.
- Dynamic metadata-dependent windows/input binding and source access bounds remain in ProgramIO.
- No output injectivity is added to the FP pair, which compares identical destination routing/scatter.

Exact FP program pairs and numerical dependencies are recorded in the [kcache_copy_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
