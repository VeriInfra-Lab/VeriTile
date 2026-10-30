import bench.pretrain_kernels.tasks.te_fused_attn_mla_thd_dv.Contract

namespace VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dv
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
    let input2 := xs.input2
    let vfill : Fin 37 → Int := fun i0 => 0
    let vslice_scatter : Fin 128 → Int := fun i0 => (if (i0.val ≥ 0 ∧ i0.val < 37 ∧ (i0.val - 0) % 1 = 0) then vfill ⟨((i0.val - 0)) % 37, Nat.mod_lt _ (by decide)⟩ else 0)
    let vfill_1 : Fin 27 → Int := fun i0 => 37
    let vslice_scatter_1 : Fin 128 → Int := fun i0 => (if (i0.val ≥ 37 ∧ i0.val < 64 ∧ (i0.val - 37) % 1 = 0) then vfill_1 ⟨((i0.val - 37)) % 27, Nat.mod_lt _ (by decide)⟩ else vslice_scatter i0)
    let vfill_2 : Fin 27 → Int := fun i0 => 64
    let vslice_scatter_2 : Fin 128 → Int := fun i0 => (if (i0.val ≥ 64 ∧ i0.val < 91 ∧ (i0.val - 64) % 1 = 0) then vfill_2 ⟨((i0.val - 64)) % 27, Nat.mod_lt _ (by decide)⟩ else vslice_scatter_1 i0)
    let vfill_3 : Fin 37 → Int := fun i0 => 91
    let vslice_scatter_3 : Fin 128 → Int := fun i0 => (if (i0.val ≥ 91 ∧ i0.val < 128 ∧ (i0.val - 91) % 1 = 0) then vfill_3 ⟨((i0.val - 91)) % 37, Nat.mod_lt _ (by decide)⟩ else vslice_scatter_2 i0)
    let veq : Fin 128 → Fin 128 → Bool := fun i0 i1 => decide (vslice_scatter_3 i0 = vslice_scatter_3 i1)
    let vge : Fin 128 → Fin 128 → Bool := fun i0 i1 => decide ((0 + (i0.val : Int) * 1) ≥ (0 + (i1.val : Int) * 1))
    let vbitwise_and : Fin 128 → Fin 128 → Bool := fun i0 i1 => (veq i0 i1 && vge i0 i1)
    let vbitwise_not : Fin 128 → Fin 128 → Bool := fun i0 i1 => (!vbitwise_and i0 i1)
    let vmasked_fill : Fin 128 → Fin 128 → ℝ := fun i0 i1 => (if vbitwise_not i0 i1 then (-338953138925153550000000000000000000000 : ℝ) else 0)
    let vbmm : Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, input0 i1 i0 lane * input1 i2 i0 lane)
    let vmul : Fin 1 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm i1 i2 i3 * (7216878364870323/100000000000000000))
    let vadd : Fin 1 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmasked_fill i2 i3)
    let vto_copy : Fin 1 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vsoftmax : Fin 1 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_1 : Fin 1 → Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vbmm_2 : Fin 4 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * input2 lane i0 i2)
    vbmm_2 i1 i0 i2)

end
end VeriTile.Bench.PretrainKernels.te_fused_attn_mla_thd_dv
