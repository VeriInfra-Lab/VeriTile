# matmul_dequantize

- Name:

gptq_kernel_correct

Statement:

Spec.Real (ReaderIO.gptq p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val))
- Name:

s2_kernel_correct

Statement:

Spec.Real (ReaderIO.s2 p valid jobs ⊨ fun xs row column =>
      ∑ k ∈ Finset.range p.inner, xs.arrays.a row.val k *
        (((Real.signedNibble (xs.arrays.weight (k/8) column.val) k -
          Real.signedNibble (xs.arrays.zero (k/p.groupSize) (column.val/8)) column.val : Int) : ℝ) *
            xs.arrays.scale (k/p.groupSize) column.val))
- Name:

dequant32_kernel_correct

Statement:

Spec.Real (ReaderIO.d32 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column))
- Name:

dequant16_kernel_correct

Statement:

Spec.Real (ReaderIO.d16 p valid jobs ⊨ fun xs row column =>
      (((Real.signedNibble (xs.weight row column) row.val -
        Real.signedNibble (xs.zero row column) column.val : Int) : ℝ) * xs.scale row column))

- [matmul_dequantize_repaired.py](matmul_dequantize_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

All four original stages remain separately specified: complete GPTQ, repaired S2, FP32 standalone dequant and FP16 standalone dequant grids, arbitrary complete job permutation, active masked outputs and exact exterior frame. External torch.mm consumer of the FP16 dequant route is not claimed.

## Conditions

- All four retain Geometry.Valid and output/input region separation; source tiled IO/output masks and ordinary binding/bounds are unchanged.
- GPTQ additionally retains blockInner=32, groupSize=32*G, noGroups=>groupSize=inner and inner=steps*32.
- S2 retains blockInner=32 and steps=(inner+31)/32, so the finite sum covers logical inner including masked partial K tile.
- The original checked metadata/index/allocation contract and route-specific dtypes/group/divisibility conditions remain untouched.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_dequantize contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
