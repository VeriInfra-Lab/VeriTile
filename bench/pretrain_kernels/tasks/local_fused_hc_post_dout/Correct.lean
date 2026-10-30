import bench.pretrain_kernels.tasks.local_fused_hc_post_dout.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_dout
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2) =>
    let input0 := xs.input0
    let input1 := xs.input1
    let vto_copy_3 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => input1 i0 i1 i2 i3
    let vmul_1 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (vto_copy_3 i0 i1 i2 i3 * input0 i0 i1 i2)
    let vsum_1 : Fin 2 → Fin 4096 → Fin 1 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 4, vmul_1 i0 i1 lane i3)
    let vto_copy_4 : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => vsum_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2
    vto_copy_4 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_dout
