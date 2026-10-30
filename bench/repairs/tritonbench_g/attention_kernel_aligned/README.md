# attention_kernel_aligned

Full logical attention softmax weighted value sum in both original FP16 and BF16 branches.

- [attention_kernel_aligned_repaired.py](attention_kernel_aligned_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete original source grids, every batch/head/query/channel output, any complete schedule permutation, safe successful execution and exact frames. Neither storage branch removed.

## Conditions

- Guard p, Nonempty p→Separated p and complete work order retained in both main theorems.
- Original configuration, strides, lengths, output injection, bias N≤64*blockN and physical machine constraints remain in Guard/IO/Contract. Pure target types use existing ChannelValues/layout, independent of source schedule.

Exact FP program pairs and numerical dependencies are recorded in the [attention_kernel_aligned contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
