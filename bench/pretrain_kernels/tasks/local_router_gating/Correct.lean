import bench.pretrain_kernels.tasks.local_router_gating.Contract

namespace VeriTile.Bench.PretrainKernels.local_router_gating
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1) =>
    let parameter0 := xs.parameter0
    let input0 := xs.input0
    let vmul : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/50))
    let vto_copy : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => input0 ⟨(((((i0.val * 2048) + i1.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1
    let vto_copy_1 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => vmul i0 i1
    let vmm : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vto_copy i0 lane * vto_copy_1 i1 lane)
    vmm i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_router_gating
