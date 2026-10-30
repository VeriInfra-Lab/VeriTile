import bench.pretrain_kernels.tasks.local_mla_fa4_hd256_absorb_reuse_sink_dsink.Correct
import bench.pretrain_kernels.tasks.local_mla_fa4_hd256_absorb_reuse_sink_dsink.FPEquiv

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_absorb_reuse_sink_dsink
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped VeriTile.Triton.LogicalScheduledKernelIO
example : correctnessProblem = Spec.Real (IO.realIO ⊨ Contract.output) := rfl
example : IO.core.nIn = 10 := rfl
example : IO.core.oarity (0 : Fin 1) = 8 := rfl
example : IO.inputs.symm ∘ IO.inputs = id := Equiv.symm_comp_self _
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_absorb_reuse_sink_dsink
