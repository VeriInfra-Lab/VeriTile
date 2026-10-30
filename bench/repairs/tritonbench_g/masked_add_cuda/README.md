# masked_add_cuda

- Name:

masked_add_kernel_correctness

Statement:

Spec.Real (ReaderIO.view storage n alpha ⊨ fun xs _ _ j => if xs.mask j = 0 then xs.gradient j + xs.data j * alpha else xs.gradient j)

- [masked_add_cuda_repaired.py](masked_add_cuda_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

The original regional kernel judgment quantifies source pid coordinates and logical block lanes with bounds masks; only zero-veto grad cells are writable. Vetoed grad cells remain in the exact exterior frame.

## Conditions

- Both int32/int64 signed-mask storage profiles and FP32 grad/data remain. n and alpha are unchanged.
- UKernelIO regional read windows, data-dependent write mask, ordinary bounds and input pinning supply the physical conditions.
- Contract retains contiguous matching shapes/devices, mask dtype choice, alpha domain and nonempty alignment/storage separation with empty bypass.

Exact FP program pairs and numerical dependencies are recorded in the [masked_add_cuda contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
