# triton_softmax

Unshifted exp(row[i]) / sum over exactly Fin n active logical columns.

- [triton_softmax_repaired.py](triton_softmax_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete actual FP32 row invocation at arbitrary source/storage offsets and row/column strides, n logical columns inside B-lane block. FP row owns each actual output thread in the fixed-four-warp native graph.

## Conditions

- 0<B; n≤B explicitly public; active output address injectivity over i<n
- FP pair: output/input row+column shape equality, y.Injective, 0<B, x.cols≤B, p<x.rows
- Contract reuses checked contiguous CUDA FP32 rank-2 domain, row grid/front-end bounds and nonempty alignment; actual launch blockSize derives coverage

Exact FP program pairs and numerical dependencies are recorded in the [triton_softmax contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
