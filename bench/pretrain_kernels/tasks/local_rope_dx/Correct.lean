import bench.pretrain_kernels.tasks.local_rope_dx.Contract

namespace VeriTile.Bench.PretrainKernels.local_rope_dx
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
    let vto_copy : Fin 64 → ℝ := fun i0 => ((0 + (i0.val : Int) * 2) : ℝ)
    let vdiv : Fin 64 → ℝ := fun i0 => (vto_copy i0 / 128)
    let vpow_1 : Fin 64 → ℝ := fun i0 => Real.rpow (10000) (vdiv i0)
    let vreciprocal : Fin 64 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul : Fin 64 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vto_copy_2 : Fin 1 → Fin 1 → Fin 128 → ℝ := fun i0 i1 i2 => ((0 + (i2.val : Int) * 1) : ℝ)
    let vbmm : Fin 1 → Fin 64 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1, vmul ⟨(((i0.val * 64) + i1.val + lane.val)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + (lane.val * 128) + i2.val)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_1 : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if i2.val < 64 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 64) + (i2.val - 0)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩ i1 else vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 64) + (i2.val - 64)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩ i1)
    let vcos : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (Real.cos (vcat_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_3 : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vcos ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vsin : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (Real.sin (vcat_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_4 : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vsin ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vto_copy_3 : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vmul_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_4 : Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vmul_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmul_9 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 i3 * vto_copy_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vneg_2 : Fin 2 → Fin 16 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (- (vmul_9 i0 i1 i2 ⟨(i3.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vslice_backward : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 64 ∧ i3.val < 128 ∧ (i3.val - 64) % 1 = 0) then vneg_2 i0 i1 i2 ⟨((i3.val - 64)) % 64, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 1 = 0) then vmul_9 i0 i1 i2 ⟨(((i3.val - 0) + 64)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_2 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vslice_backward i0 i1 i2 i3 + vslice_backward_1 i0 i1 i2 i3)
    let vmul_10 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 i3 * vto_copy_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vadd_3 : Fin 2 → Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vadd_2 i0 i1 i2 i3 + vmul_10 i0 i1 i2 i3)
    vadd_3 i0 i1 i2 i3)

end
end VeriTile.Bench.PretrainKernels.local_rope_dx
