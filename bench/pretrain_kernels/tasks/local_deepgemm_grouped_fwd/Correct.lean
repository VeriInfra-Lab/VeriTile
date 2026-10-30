import bench.pretrain_kernels.tasks.local_deepgemm_grouped_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
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
    let vmul : Fin 2 → Fin 640 → Fin 2048 → ℝ := fun i0 i1 i2 => (parameter0 i0 i1 i2 * (1/50))
    let vmm : Fin 128 → Fin 640 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, input0 ⟨(i0.val) % 256, Nat.mod_lt _ (by decide)⟩ lane * vmul ⟨(0) % 2, Nat.mod_lt _ (by decide)⟩ i1 lane)
    let vmm_1 : Fin 128 → Fin 640 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, input0 ⟨((i0.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ lane * vmul ⟨(1) % 2, Nat.mod_lt _ (by decide)⟩ i1 lane)
    let vcat : Fin 256 → Fin 640 → ℝ := fun i0 i1 => (if i0.val < 128 then vmm ⟨((i0.val - 0)) % 128, Nat.mod_lt _ (by decide)⟩ i1 else vmm_1 ⟨((i0.val - 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1)
    vcat i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_fwd
