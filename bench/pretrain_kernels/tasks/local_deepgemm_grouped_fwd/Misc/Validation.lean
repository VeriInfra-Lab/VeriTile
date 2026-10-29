import bench.pretrain_kernels.tasks.local_deepgemm_grouped_fwd.Correct
import bench.pretrain_kernels.tasks.local_deepgemm_grouped_fwd.FPEquiv

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped VeriTile.Triton.LogicalScheduledKernelIO
example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 2 := rfl
example : IO.core.oarity (0 : Fin 1) = 163840 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
