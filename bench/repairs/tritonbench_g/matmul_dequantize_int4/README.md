# matmul_dequantize_int4

- Name:

correct

Statement:

Spec.Real (ReaderIO.view c i ⊨{ReaderIO.packedInt32 c} fun xs row column =>
      ∑ k ∈ Finset.range c.K, xs.arrays.A row.val k *
        ((Real.nibble (xs.arrays.Q (k / 8) column.val) k -
          Real.nibble (xs.arrays.Z (k / c.group) (column.val / 8)) column.val) *
            xs.arrays.S (k / c.group) column.val))

- [matmul_dequantize_int4_repaired.py](matmul_dequantize_int4_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete grouped grid for all 17 original tunings, every logical M*N result, arbitrary original valid output strides and exact exterior frame.

## Conditions

- RawKernel.Layout.Valid remains public, including original K/group multiples, shape/stride/layout/capacity/index requirements.
- IOData.packedInt32 is retained for signed-int32 Q/Z input tensors; ordinary typed input binding/access bounds and named-region separation remain.
- FP16 A/scales/output and FP32 accumulation are the selected source profile.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_dequantize_int4 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
