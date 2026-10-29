import bench.pretrain_kernels.tasks.local_cross_entropy_ignore_dx.Contract

namespace VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs (i0,i1,i2) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let input0 := xs.input0
    let input1 := xs.input1
    let vge : Fin 128 → Fin 2 → Bool := fun i0 i1 => decide (parameter1 i0 i1 ≥ (1/8))
    let vwhere : Fin 128 → Fin 2 → Int := fun i0 i1 => (if vge i0 i1 then parameter0 i0 i1 else (-100))
    let vwhere_1 : Fin 128 → Fin 2 → Int := fun i0 i1 => (if vge i0 i1 then vwhere i0 i1 else 0)
    let vto_copy : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => input0 i0 i1 i2
    let vmax_1_value : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => ((List.finRange 1024).map (fun lane => vto_copy i0 i1 ⟨(lane.val) % 1024, Nat.mod_lt _ (by decide)⟩)).foldl max (vto_copy i0 i1 ⟨(0) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vmax_1_index : Fin 128 → Fin 2 → Fin 1 → Int := fun i0 i1 i2 => ((((List.finRange 1024).mergeSort (fun left right => decide (vto_copy i0 i1 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ > vto_copy i0 i1 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∨ (vto_copy i0 i1 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ = vto_copy i0 i1 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).headD 0).val : Int)
    let vsub : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 - vmax_1_value i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vexp : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (Real.exp (vsub i0 i1 i2))
    let vsum_1 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (∑ lane : Fin 1024, vexp i0 i1 lane)
    let vto_copy_1 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => input1 i0 i1
    let vwhere_3 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (if vge i0 i1 then vto_copy_1 i0 i1 else 0)
    let vneg_1 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (- (vwhere_3 i0 i1))
    let vneg_2 : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (- (vneg_1 i0 i1))
    let vsub_3 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 - vmax_1_value i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vexp_1 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (Real.exp (vsub_3 i0 i1 i2))
    let vdiv : Fin 128 → Fin 2 → ℝ := fun i0 i1 => (vneg_2 i0 i1 / vsum_1 i0 i1)
    let vmul : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vdiv i0 i1 * vexp_1 i0 i1 i2)
    let vneg_3 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (- (vmul i0 i1 i2))
    let vsum_2 : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1024, vneg_3 i0 i1 lane)
    let vadd : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => (vneg_2 i0 i1 + vsum_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vneg_4 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (- (vneg_3 i0 i1 i2))
    let vscatter_add : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => (0 + (∑ lane : Fin 1, if 0 = (i2.val : Int) then vneg_1 ⟨(((((i0.val * 2) + i1.val + lane.val) / 2) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 2) + i1.val + lane.val) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ else 0))
    let vgather_1 : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => vneg_4 i0 i1 ⟨(Int.toNat (vwhere_1 i0 ⟨((i1.val + i2.val)) % 2, Nat.mod_lt _ (by decide)⟩)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vadd_1 : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => (vscatter_add i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + vgather_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vadd_2 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vneg_4 i0 i1 i2 + (0 : ℝ))
    let vscatter : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => ((List.finRange 1).foldl (fun value lane => if vwhere_1 ⟨(((((i0.val * 2) + i1.val + lane.val) / 2) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 2) + i1.val + lane.val) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ = (i2.val : Int) then vadd_1 i0 i1 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ else value) (vadd_2 i0 i1 i2))
    let vgather_2 : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => vscatter i0 i1 ⟨(Int.toNat (vmax_1_index i0 i1 i2)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vadd_3 : Fin 128 → Fin 2 → Fin 1 → ℝ := fun i0 i1 i2 => (vgather_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + vadd i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vadd_4 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => (vscatter i0 i1 i2 + (0 : ℝ))
    let vscatter_1 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => ((List.finRange 1).foldl (fun value lane => if vmax_1_index i0 i1 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ = (i2.val : Int) then vadd_3 i0 i1 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ else value) (vadd_4 i0 i1 i2))
    let vto_copy_2 : Fin 128 → Fin 2 → Fin 1024 → ℝ := fun i0 i1 i2 => vscatter_1 i0 i1 i2
    vto_copy_2 i0 i1 i2))

end
end VeriTile.Bench.PretrainKernels.local_cross_entropy_ignore_dx
