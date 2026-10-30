import bench.pretrain_kernels.tasks.local_fused_hc_post_bda_dres.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dres
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2,i3) =>
    let input0 := xs.input0
    let vto_copy_5 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => input0 i0 i1 i2 i3
    let vto_copy_6 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => vto_copy_5 i0 i1 i2 i3
    vto_copy_6 i0 i1 i2 i3)

end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_bda_dres
