import bench.pretrain_kernels.tasks.local_rmsnorm_dx.Contract

namespace VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
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
    let input0 := xs.input0
    let input1 := xs.input1
    let vto_copy : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => input0 i0 i1 i2
    let vpow_1 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => ((vto_copy i0 i1 i2) * (vto_copy i0 i1 i2))
    let vmean : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 2048, vpow_1 i0 i1 lane) / 2048)
    let vadd : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_2 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => input1 i0 i1 i2
    let vmul_2 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vto_copy_2 i0 i1 i2 * parameter0 i2)
    let vmul_3 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vmul_2 i0 i1 i2 * vto_copy i0 i1 i2)
    let vmul_4 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vmul_2 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 2048, vmul_3 i0 i1 lane)
    let vpow_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_5 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_6 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_5 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vmul_6 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 2048)
    let vpow_3 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => ((vto_copy i0 i1 i2))
    let vmul_7 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vpow_3 i0 i1 i2 * (2 : ℝ))
    let vmul_8 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vdiv i0 i1 i2 * vmul_7 i0 i1 i2)
    let vadd_1 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => (vmul_4 i0 i1 i2 + vmul_8 i0 i1 i2)
    let vto_copy_3 : Fin 2 → Fin 128 → Fin 2048 → ℝ := fun i0 i1 i2 => vadd_1 i0 i1 i2
    vto_copy_3 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_rmsnorm_dx
