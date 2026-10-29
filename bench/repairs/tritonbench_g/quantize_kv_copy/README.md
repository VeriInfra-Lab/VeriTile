# quantize_kv_copy

mathematical

- [quantize_kv_copy_repaired.py](quantize_kv_copy_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete flattened schedule range n, decoded to token=pid/heads and head=pid%heads; each public n entry carries p.groups groups of eight. Host bindings instantiate n for the full token/head grid. Scatter output addresses remain KernelGridResult.target p dest.

## Conditions

groups<=block and target injectivity on pid<n retained; Metadata binds signed-int32 destination values. Public IO supplies safety/success, complete output masks and both exterior frames. Contract positive heads, width divisible by eight, groupBlock and scatter uniqueness/bounds imply admitted coverage.

Exact FP program pairs and numerical dependencies are recorded in the [quantize_kv_copy contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
