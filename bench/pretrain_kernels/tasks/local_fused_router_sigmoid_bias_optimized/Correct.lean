import bench.pretrain_kernels.tasks.local_fused_router_sigmoid_bias_optimized.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_router_sigmoid_bias_optimized
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
    let vmul : Fin 16 → Int := fun i0 => ((0 + (i0.val : Int) * 1) * 17)
    let vadd : Fin 16 → Int := fun i0 => (vmul i0 + 7)
    let vremainder : Fin 16 → Int := fun i0 => (vadd i0 % 128)
    let vindex_put : Fin 128 → ℝ := fun i0 => (List.finRange 16).foldl (fun acc lane => if vremainder ⟨(lane.val) % 16, Nat.mod_lt _ (by decide)⟩ = (i0.val : Int) then ((1/20) + (lane.val : ℝ) * (1/500)) else acc) (0)
    let vsigmoid : Fin 17 → Fin 128 → ℝ := fun i0 i1 => (1 / (1 + Real.exp (-(input0 i0 i1))))
    let vadd_1 : Fin 17 → Fin 128 → ℝ := fun i0 i1 => (vsigmoid i0 i1 + vindex_put i1)
    let visnan : Fin 17 → Fin 128 → Bool := fun i0 i1 => decide (False)
    let vwhere : Fin 17 → Fin 128 → ℝ := fun i0 i1 => (if visnan i0 i1 then literal_nan ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else vadd_1 i0 i1)
    let vwhereFinite : Fin 17 → Fin 128 → Prop := fun i0 i1 => (if visnan i0 i1 then False else True)
    let vsort_value : Fin 17 → Fin 128 → ℝ := fun i0 i1 => vwhere i0 ⟨((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 128, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 17 → Fin 128 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vgather_1 : Fin 17 → Fin 16 → ℝ := fun i0 i1 => vsigmoid i0 ⟨(Int.toNat (vsort_index i0 ⟨(i1.val) % 128, Nat.mod_lt _ (by decide)⟩)) % 128, Nat.mod_lt _ (by decide)⟩
    let vsum_1 : Fin 17 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 16, vgather_1 i0 lane)
    let vadd_2 : Fin 17 → Fin 1 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/100000000000000000000))
    let vdiv : Fin 17 → Fin 16 → ℝ := fun i0 i1 => (vgather_1 i0 i1 / vadd_2 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_1 : Fin 17 → Fin 16 → ℝ := fun i0 i1 => (vdiv i0 i1 * (1413/500))
    let vscatter : Fin 17 → Fin 128 → ℝ := fun i0 i1 => ((List.finRange 16).foldl (fun value lane => if vsort_index i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then vmul_1 i0 ⟨(lane.val) % 16, Nat.mod_lt _ (by decide)⟩ else value) (0))
    vscatter i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_fused_router_sigmoid_bias_optimized
