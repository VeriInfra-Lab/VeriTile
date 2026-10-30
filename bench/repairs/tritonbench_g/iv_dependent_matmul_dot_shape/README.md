# iv_dependent_matmul

kernel_real: C[row,col] = sum over Fin g.inner of A[row,k]*B[k,col].

- [iv_dependent_matmul_dot_shape.py](iv_dependent_matmul_dot_shape.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Complete selected source grid for each retained pointer schedule; source graph has Dot+Add and ideal Real interpretation, native graph retains its live FP32 carry, TF32 K8 atoms, FP16 narrowing and FP32 stores.

## Conditions

- Original IO.Layout (logical/physical output separation, alignment and addressability) and Index64.Launchable request remain explicit.
- Checked contract retains live FP32 dot minimum shape 16≤blockInner; empty branches remain admitted.
- No new desired output, source run or caller-provided execution premise.

Exact FP program pairs and numerical dependencies are recorded in the [iv_dependent_matmul contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
