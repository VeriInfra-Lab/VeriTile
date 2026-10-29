import bench.pretrain_kernels.tasks.local_fused_aux_loss_topk6_odd.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_topk6_odd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs i0 =>
    let input0 := xs.input0
    let input1 := xs.input1
    let configurationSmCount := xs.configurationSmCount
    let vsum_1 : Fin 128 → ℝ := fun i0 => (∑ lane : Fin 513, input0 lane i0)
    let vmul : Fin 128 → ℝ := fun i0 => (vsum_1 i0 * input1 i0)
    let vsum_2 : Fin 1 → ℝ := fun i0 => (∑ lane : Fin 128, vmul lane)
    let vmul_1 : Fin 1 → ℝ := fun i0 => (vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (8106324579769401/10000000000000000000000))
    vmul_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))

end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_topk6_odd
