# adam_update_triton

Independent Lion parameter decay/sign update and next momentum.

- [adam_update_triton_repaired.py](adam_update_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One actual Real invocation at arbitrary pid, active pid*B+j<n, both in-place P/M outputs, safe successful execution and exact exterior frame. The FP result separately covers the complete grid.

## Conditions

- Runtime lr/wd/beta1/beta2 register bindings, P/G/M finite input windows and output/access bounds are in Interface.programIO/IO.lean; no literal scalar substitution.
- Contract binds contiguous FP32 P/G/M, normalized runtime scalars, n/total/chunk geometry, disjoint live roles, pointer/empty-launch conditions.

Exact FP program pairs and numerical dependencies are recorded in the [adam_update_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
