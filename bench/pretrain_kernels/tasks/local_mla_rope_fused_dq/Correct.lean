import bench.pretrain_kernels.tasks.local_mla_rope_fused_dq.Contract

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dq
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
    let vto_copy : Fin 4 → ℝ := fun i0 => ((0 + (i0.val : Int) * 2) : ℝ)
    let vdiv : Fin 4 → ℝ := fun i0 => (vto_copy i0 / 8)
    let vpow_1 : Fin 4 → ℝ := fun i0 => Real.rpow (10000) (vdiv i0)
    let vreciprocal : Fin 4 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul : Fin 4 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vto_copy_4 : Fin 1 → Fin 1 → Fin 128 → ℝ := fun i0 i1 i2 => ((0 + (i2.val : Int) * 1) : ℝ)
    let vbmm : Fin 1 → Fin 4 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1, vmul ⟨(((i0.val * 4) + i1.val + lane.val)) % 4, Nat.mod_lt _ (by decide)⟩ * vto_copy_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + (lane.val * 128) + i2.val)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_2 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (if i2.val < 4 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 4) + (i2.val - 0)) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i1 else vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 4) + (i2.val - 4)) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i1)
    let vcos_1 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (Real.cos (vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_5 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (vcos_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (5/4))
    let vsin_1 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (Real.sin (vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_6 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (vsin_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (5/4))
    let vto_copy_5 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => vmul_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_6 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => vmul_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmul_15 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨((i3.val + 20)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vmul_16 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨((i3.val + 20)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vneg : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (- (input0 i0 i1 i2 ⟨((i3.val + 16)) % 64, Nat.mod_lt _ (by decide)⟩))
    let vmul_17 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vneg i0 i1 i2 i3 * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vadd_2 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_16 i0 i1 i2 i3 + vmul_17 i0 i1 i2 i3)
    let vmul_18 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨((i3.val + 16)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vadd_3 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_15 i0 i1 i2 i3 + vmul_18 i0 i1 i2 i3)
    let vslice_backward : Fin 2 → Fin 12 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 8 ∧ (i3.val - 1) % 2 = 0) then vadd_2 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 4, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 2 → Fin 12 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 8 ∧ (i3.val - 0) % 2 = 0) then vadd_3 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 4, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_4 : Fin 2 → Fin 12 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 i3 => (vslice_backward i0 i1 i2 i3 + vslice_backward_1 i0 i1 i2 i3)
    let vcat_8 : Fin 2 → Fin 12 → Fin 128 → Fin 24 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 16 then input0 i0 i1 i2 ⟨((i3.val - 0)) % 64, Nat.mod_lt _ (by decide)⟩ else vadd_4 i0 i1 i2 ⟨((i3.val - 16)) % 8, Nat.mod_lt _ (by decide)⟩)
    vcat_8 i0 i1 i2 i3)

end
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused_dq
