import bench.pretrain_kernels.tasks.local_fused_router.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_router
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (_xs : IO.Inputs) : Prop :=
  True

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let input0 := xs.input0
    let literal_nan : Fin 1 → ℝ := fun _ => 0
    let vsigmoid : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (1 / (1 + Real.exp (-(input0 i0 i1))))
    let visnan : Fin 256 → Fin 128 → Bool := fun i0 i1 => decide (False)
    let vwhere : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (if visnan i0 i1 then literal_nan ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else vsigmoid i0 i1)
    let vwhereFinite : Fin 256 → Fin 128 → Prop := fun i0 i1 => (if visnan i0 i1 then False else True)
    let vsort_value : Fin 256 → Fin 128 → ℝ := fun i0 i1 => vwhere i0 ⟨((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 128, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 256 → Fin 128 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vgather : Fin 256 → Fin 6 → ℝ := fun i0 i1 => vsigmoid i0 ⟨(Int.toNat (vsort_index i0 ⟨(i1.val) % 128, Nat.mod_lt _ (by decide)⟩)) % 128, Nat.mod_lt _ (by decide)⟩
    let vsum_1 : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 6, vgather i0 lane)
    let vadd : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/100000000000000000000))
    let vdiv : Fin 256 → Fin 6 → ℝ := fun i0 i1 => (vgather i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul : Fin 256 → Fin 6 → ℝ := fun i0 i1 => (vdiv i0 i1 * (1413/500))
    let vscatter : Fin 256 → Fin 128 → ℝ := fun i0 i1 => ((List.finRange 6).foldl (fun value lane => if vsort_index i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then vmul i0 ⟨(lane.val) % 6, Nat.mod_lt _ (by decide)⟩ else value) (0))
    vscatter i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_router
