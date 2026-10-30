import bench.pretrain_kernels.tasks.te_fused_attn_mla_dv.Correct
import bench.pretrain_kernels.tasks.te_fused_attn_mla_dv.FPEquiv

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dv
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped VeriTile.Triton.LogicalScheduledKernelIO
example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 3 := rfl
example : IO.core.oarity (0 : Fin 1) = 131072 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_dv
