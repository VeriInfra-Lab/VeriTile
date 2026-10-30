# quantize_kv_transform

kernel_real

- [quantize_kv_transform_repaired.py](quantize_kv_transform_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Both complete code/FP32-scale scattered outputs for every token/head/logical column, under any order.Perm(range count), including count=0.

## Conditions

0<columns, heads<=height, columns<=width and exact schedule coverage retained. Destinations requires nonnegative signed-int32 metadata and injectivity; allocation bounds remain in public IO. Launch parameters are headBlock/columnBlock and existing coverage lemmas establish them from the Contract.

Exact FP program pairs and numerical dependencies are recorded in the [quantize_kv_transform contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
