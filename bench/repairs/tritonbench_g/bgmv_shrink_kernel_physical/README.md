# bgmv_shrink_kernel

Initial Y plus scaling times the complete logical K gathered GEMV; signed index -1 preserves initial Y. Existing toFloat/ofFloat wrappers retain the declared channel/storage interpretation.

- [bgmv_shrink_kernel_repaired.py](bgmv_shrink_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full batches by 64 launch catalog and every address's independently complete atomic event order; all N outputs, sentinel rows, K/N empty cases and original FP16/BF16 storage choices.

## Conditions

Public conditions:

- Call retains source shape, storage and output conversion metadata
- Orders h records the original per-address atomic order/completeness
- IO.domain binds actual metadata cells to indices and requires -1 or nonnegative signed indices
- PerAddressProgramIO keeps all-program success/safety, locality, coverage, exact tagged cell binding, outputs and dual frames

Contract domain source:

Contract retains the checked tensor/shape/alignment/separation/signed-metadata domain. IO.slots covers every selected adapter slot without adding an adapter-count guard to the kernel theorem; IO.inputMask still reads X on active CTAs even when N=0.

Exact FP program pairs and numerical dependencies are recorded in the [bgmv_shrink_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
