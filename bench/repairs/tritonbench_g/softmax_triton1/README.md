# softmax_triton1

exp(xs i) / sum (j : Fin n), exp(Logical.active cover xs j)

- [softmax_triton1_repaired.py](softmax_triton1_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One actual contiguous active row launch with caller-selected base offsets; all n logical columns are covered when n≤B. The public masked IO proves flattening, safe accesses under IO bounds, successful execution, all active output cells and the exterior frame. This is not a newly proved complete host/grid/Tensor application.

## Conditions

- 0<B
- n≤B is now explicit in the main theorem
- MaskedKernelIO₁ input/output region binding and bounds remain in the public judgment
- Original no-cover masked-Fin-B window theorem remains available as RealIO.softmax_window_correctness in Proofs/RealIO.lean
- Existing Contract blockSize and Proofs/Contract.blockSize_bounds derive coverage from launch configuration, so no Python/domain restriction was added

Exact FP program pairs and numerical dependencies are recorded in the [softmax_triton1 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
