# embedding_triton_kernel

For each row/feature, gather weights[id-first, feature] when first≤id<last, otherwise zero.

- [embedding_triton_kernel_repaired.py](embedding_triton_kernel_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

One complete invocation visiting 64 rows pid*64+a; active rows require <n and columns d<hidden. Width is the masked tile extent; the vocabulary window is (last-first).toNat. Only selected active weight rows are bound/read.

## Conditions

- Execution.Layout explicitly requires ids/output and weight/output separation, hidden≤width and hidden≤outputStride.
- IO.signed restricts only active read metadata to signed int32; payload Real values are unrestricted.
- Public IO exposes all active initial bounds/bindings and all output/frame conclusions. Contract maps checked first/last and vocabulary capacity into these parameters without adding a desired-output premise.

Exact FP program pairs and numerical dependencies are recorded in the [embedding_triton_kernel contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
