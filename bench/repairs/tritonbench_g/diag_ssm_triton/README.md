# diag_ssm_triton

Forward time recurrence and reverse carried/input/initial/transition gradients for FP32-real and complex64 mathematical interpretations.

- [diag_ssm_triton_repaired.py](diag_ssm_triton_repaired.py)
- [Kernels.lean](Kernels.lean)
- [Contract.lean](Contract.lean)
- [Correct.lean](Correct.lean)
- [FPEquiv.lean](FPEquiv.lean)

## Scope

Four existing per-program regional judgments cover each complete time loop. Logical lanes are pid*B+lane<batch*dim. Forward outputs are length*B; backward gx is length*B, gs/gl are B. gl remains a per-batch/column gradient, with no later host batch reduction.

## Conditions

- Real forward retains p.x≠p.y; real backward retains RegionCorrectness.BackwardSeparation; complex backward retains ComplexBackwardRun.Separation; complex forward retains its existing alias support.
- Regional IO quantifies the actual pid/context, coherent initial logical bindings, masked storage bounds and permitted read aliases; access safety, execution, outputs and exterior frame are conclusions.
- Contract metadata supports FP32 and complex64, same devices/dtypes and exact [time,batch,dim] shape relationships; the headline is the explicitly described kernel-stage scope.

Exact FP program pairs and numerical dependencies are recorded in the [diag_ssm_triton contract review](../../../validation/contracts.json). The final source inventory, Lean build, axiom/spec audits and independent proof replay are separate checks; this review does not certify GPU execution.
