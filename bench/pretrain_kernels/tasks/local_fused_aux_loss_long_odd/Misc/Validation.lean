import bench.pretrain_kernels.tasks.local_fused_aux_loss_long_odd.Correct
import bench.pretrain_kernels.tasks.local_fused_aux_loss_long_odd.FPEquiv

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_odd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped VeriTile.Triton.LogicalScheduledKernelIO
example : correctnessProblem = Spec.Real (IO.realIO.ImplementsOn Contract.Allowed Contract.output) := rfl
example : IO.core.nIn = 3 := rfl
example : IO.core.oarity (0 : Fin 1) = 1 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_odd
