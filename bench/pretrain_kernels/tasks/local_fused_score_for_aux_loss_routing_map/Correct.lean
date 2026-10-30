import bench.pretrain_kernels.tasks.local_fused_score_for_aux_loss_routing_map.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1) =>
    let input0 := xs.input0
    let literal_nan : Fin 1 → ℝ := fun _ => 0
    let vsigmoid : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (1 / (1 + Real.exp (-(input0 i0 i1))))
    let vsum_1 : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 128, vsigmoid i0 lane)
    let vadd : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/100000000000000000000))
    let vdiv : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (vsigmoid i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let visnan : Fin 256 → Fin 128 → Bool := fun i0 i1 => decide (False)
    let vwhere : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (if visnan i0 i1 then literal_nan ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else vdiv i0 i1)
    let vwhereFinite : Fin 256 → Fin 128 → Prop := fun i0 i1 => (if visnan i0 i1 then False else True)
    let vsort_value : Fin 256 → Fin 128 → ℝ := fun i0 i1 => vwhere i0 ⟨((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 128, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 256 → Fin 128 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vto_copy : Fin 256 → Fin 128 → Int := fun i0 i1 => 0
    let vscatter : Fin 256 → Fin 128 → Int := fun i0 i1 => ((List.finRange 16).foldl (fun value lane => if vsort_index i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then 1 else value) (vto_copy i0 i1))
    let vto_copy_1 : Fin 256 → Fin 128 → Bool := fun i0 i1 => decide (vscatter i0 i1 ≠ 0)
    let vto_copy_2 : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (if vto_copy_1 i0 i1 then 1 else 0)
    vto_copy_2 i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss_routing_map
