import bench.pretrain_kernels.tasks.local_mlp_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_mlp_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let input0 := xs.input0
    let vmul : Fin 6144 → Fin 2048 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/50))
    let vmul_1 : Fin 6144 → Fin 2048 → ℝ := fun i0 i1 => (parameter1 i0 i1 * (1/50))
    let vmul_2 : Fin 2048 → Fin 6144 → ℝ := fun i0 i1 => (parameter2 i0 i1 * (1/50))
    let vmm : Fin 256 → Fin 6144 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, input0 ⟨(((((i0.val * 2048) + lane.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vmul i1 lane)
    let vmm_1 : Fin 256 → Fin 6144 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, input0 ⟨(((((i0.val * 2048) + lane.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vmul_1 i1 lane)
    let vto_copy : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vsilu : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 * (1 / (1 + Real.exp (-(vto_copy i0 i1 i2)))))
    let vto_copy_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vmm_1 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vmul_3 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vsilu i0 i1 i2 * vto_copy_1 i0 i1 i2)
    let vto_copy_2 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vmul_3 i0 i1 i2
    let vmm_2 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 6144, vto_copy_2 ⟨(((((i0.val * 6144) + lane.val) / 786432) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vmul_2 i1 lane)
    vmm_2 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2)

end
end VeriTile.Bench.PretrainKernels.local_mlp_fwd
