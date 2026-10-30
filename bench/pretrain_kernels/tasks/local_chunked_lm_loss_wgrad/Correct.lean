import bench.pretrain_kernels.tasks.local_chunked_lm_loss_wgrad.Contract

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO.ImplementsOn Contract.Allowed (fun xs (i0,i1) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let input0 := xs.input0
    let input1 := xs.input1
    let vmul : Fin 1024 → Fin 64 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/20))
    let vgt : Fin 320 → Bool := fun i0 => decide (parameter2 i0 > (3/10))
    let vto_copy : Fin 320 → ℝ := fun i0 => (if vgt i0 then 1 else 0)
    let vmm : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 64, input0 i0 lane * vmul i1 lane)
    let vto_copy_1 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => vmm i0 i1
    let vmax_1_value : Fin 320 → Fin 1 → ℝ := fun i0 i1 => ((List.finRange 1024).map (fun lane => vto_copy_1 i0 ⟨(lane.val) % 1024, Nat.mod_lt _ (by decide)⟩)).foldl max (vto_copy_1 i0 ⟨(0) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vmax_1_index : Fin 320 → Fin 1 → Int := fun i0 i1 => ((((List.finRange 1024).mergeSort (fun left right => decide (vto_copy_1 i0 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ > vto_copy_1 i0 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∨ (vto_copy_1 i0 ⟨(left.val) % 1024, Nat.mod_lt _ (by decide)⟩ = vto_copy_1 i0 ⟨(right.val) % 1024, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).headD 0).val : Int)
    let vsub : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (vto_copy_1 i0 i1 - vmax_1_value i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vexp : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (Real.exp (vsub i0 i1))
    let vsum_1 : Fin 320 → ℝ := fun i0 => (∑ lane : Fin 1024, vexp i0 lane)
    let vto_copy_2 : Fin 1 → ℝ := fun i0 => input1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩
    let vmul_2 : Fin 320 → ℝ := fun i0 => (vto_copy_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vto_copy i0)
    let vneg_1 : Fin 320 → ℝ := fun i0 => (- (vmul_2 i0))
    let vneg_2 : Fin 320 → ℝ := fun i0 => (- (vneg_1 i0))
    let vsub_3 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (vto_copy_1 i0 i1 - vmax_1_value i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vexp_1 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (Real.exp (vsub_3 i0 i1))
    let vdiv : Fin 320 → ℝ := fun i0 => (vneg_2 i0 / vsum_1 i0)
    let vmul_3 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (vdiv i0 * vexp_1 i0 i1)
    let vneg_3 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (- (vmul_3 i0 i1))
    let vsum_4 : Fin 320 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 1024, vneg_3 i0 lane)
    let vadd : Fin 320 → Fin 1 → ℝ := fun i0 i1 => (vneg_2 i0 + vsum_4 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vneg_4 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (- (vneg_3 i0 i1))
    let vscatter_add : Fin 320 → Fin 1 → ℝ := fun i0 i1 => (0 + (∑ lane : Fin 1, if 0 = (i1.val : Int) then vneg_1 ⟨(((i0.val + lane.val) % 320)) % 320, Nat.mod_lt _ (by decide)⟩ else 0))
    let vgather_1 : Fin 320 → Fin 1 → ℝ := fun i0 i1 => vneg_4 i0 ⟨(Int.toNat (parameter1 ⟨((i0.val + i1.val)) % 320, Nat.mod_lt _ (by decide)⟩)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vadd_1 : Fin 320 → Fin 1 → ℝ := fun i0 i1 => (vscatter_add i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + vgather_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vadd_2 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (vneg_4 i0 i1 + (0 : ℝ))
    let vscatter : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => ((List.finRange 1).foldl (fun value lane => if parameter1 ⟨(((i0.val + lane.val) % 320)) % 320, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then vadd_1 i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ else value) (vadd_2 i0 i1))
    let vgather_2 : Fin 320 → Fin 1 → ℝ := fun i0 i1 => vscatter i0 ⟨(Int.toNat (vmax_1_index i0 i1)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vadd_3 : Fin 320 → Fin 1 → ℝ := fun i0 i1 => (vgather_2 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vadd_4 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => (vscatter i0 i1 + (0 : ℝ))
    let vscatter_1 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => ((List.finRange 1).foldl (fun value lane => if vmax_1_index i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then vadd_3 i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ else value) (vadd_4 i0 i1))
    let vto_copy_3 : Fin 320 → Fin 1024 → ℝ := fun i0 i1 => vscatter_1 i0 i1
    let vmm_1 : Fin 1024 → Fin 64 → ℝ := fun i0 i1 => (∑ lane : Fin 320, vto_copy_3 lane i0 * input0 lane i1)
    vmm_1 i0 i1))

end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_wgrad
