# matmul_leakyrelu_fp8

- Name:

kernel_correct

Statement:

VeriTile.Spec.Real (ReaderIO.view M g b ⊨ fun xs r c =>
      let value

- [matmul_leakyrelu_fp8_scalar_cast.py](matmul_leakyrelu_fp8_scalar_cast.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected grouped source grid, K=0 and both activation branches, every logical output and exact exterior frame; selected source actually reads raw FP32 despite the historical fp8 task name.

## Conditions

- Execution.ScalarLaws M realOps and KernelBuffers.Layout g b remain public. These bind the explicit selected scalar model/source IO and physical layout, not a desired output.
- Original geometry, arbitrary valid view strides, input/output region separation, typed binding and bounds remain; no native positiveK auxiliary condition is added to Real correctness.
- Contract retains signed ABI/storage metadata and actual Tensor alias reader scope.

Exact FP program pairs and numerical dependencies are recorded in the [matmul_leakyrelu_fp8 contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
