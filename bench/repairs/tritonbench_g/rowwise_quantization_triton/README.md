# rowwise_quantization_triton

m=max_{0≤c<columns}|x[row,c]| (empty maximum zero); byte=low8(roundEven(127*x/m)), or zero when m=0; second output is m under Real erasure of FP16 float cast.

- [rowwise_quantization_triton_repaired.py](rowwise_quantization_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Full row permutation from Schedule p, all logical columns and both output channels. Original reduction tree has exactly width leaves with columns≤width; padding contributes zero. INT8 representation and FP16 maximum conversion remain explicit in native IO.

## Conditions

- Launch: positive columns, columns≤width, tree.leaves.Perm (range width). Schedule covers a permutation of every row.
- IO.byteView selects the explicit integer byte interpretation; this specialized observation is retained.
- The public IO judgment quantifies over logical inputs, physical memory binding and capacity bounds; source safety, successful execution, output values and exact exterior frame are conclusions, not caller premises.

Exact FP program pairs and numerical dependencies are recorded in the [rowwise_quantization_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
