import bench.pretrain_kernels.tasks.local_masked_moe_aux_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_masked_moe_aux_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (∀ (j0 : Fin 4), xs.constantf1ebabb642 j0 = ([1, 3, 0, 6] : List Int).getD (j0.val) 0)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let constantf1ebabb642 := xs.constantf1ebabb642
    let input0 := xs.input0
    let literal_nan : Fin 1 → ℝ := fun _ => 0
    let vge : Fin 4 → Fin 8 → Bool := fun i0 i1 => decide ((0 + (i1.val : Int) * 1) ≥ constantf1ebabb642 i0)
    let vto_copy : Fin 32 → Fin 16 → ℝ := fun i0 i1 => input0 i0 i1
    let vsigmoid : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (1 / (1 + Real.exp (-(vto_copy i0 i1))))
    let vsum_1 : Fin 32 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 16, vsigmoid i0 lane)
    let vadd : Fin 32 → Fin 1 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/100000000000000000000))
    let vdiv : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vsigmoid i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let visnan : Fin 32 → Fin 16 → Bool := fun i0 i1 => decide (False)
    let vwhere : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (if visnan i0 i1 then literal_nan ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else vdiv i0 i1)
    let vwhereFinite : Fin 32 → Fin 16 → Prop := fun i0 i1 => (if visnan i0 i1 then False else True)
    let vsort_value : Fin 32 → Fin 16 → ℝ := fun i0 i1 => vwhere i0 ⟨((((List.finRange 16).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 16, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 16, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 16, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 16, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 16, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 32 → Fin 16 → Int := fun i0 i1 => ((((List.finRange 16).mergeSort (fun left right => decide (vwhere i0 ⟨(left.val) % 16, Nat.mod_lt _ (by decide)⟩ > vwhere i0 ⟨(right.val) % 16, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere i0 ⟨(left.val) % 16, Nat.mod_lt _ (by decide)⟩ = vwhere i0 ⟨(right.val) % 16, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vto_copy_1 : Fin 32 → Fin 16 → Int := fun i0 i1 => 0
    let vscatter : Fin 32 → Fin 16 → Int := fun i0 i1 => ((List.finRange 2).foldl (fun value lane => if vsort_index i0 ⟨(lane.val) % 16, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then 1 else value) (vto_copy_1 i0 i1))
    let vto_copy_2 : Fin 32 → Fin 16 → Bool := fun i0 i1 => decide (vscatter i0 i1 ≠ 0)
    let vbitwise_not : Fin 32 → Bool := fun i0 => (!vge ⟨(((i0.val / 8) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩)
    let vbitwise_and : Fin 32 → Fin 16 → Bool := fun i0 i1 => (vto_copy_2 i0 i1 && vbitwise_not i0)
    let vmul : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vdiv i0 i1 * (if vbitwise_not i0 then 1 else 0))
    let vsum_2 : Fin 16 → Int := fun i0 => (∑ lane : Fin 32, (if vbitwise_and lane i0 then (1 : Int) else 0))
    let vsum_3 : Fin 1 → Int := fun i0 => (∑ lane : Fin 16, vsum_2 lane)
    let vfloor_divide : Fin 1 → Int := fun i0 => (vsum_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 2)
    let vclamp_min : Fin 1 → Int := fun i0 => max (vfloor_divide ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) (1)
    let vsum_4 : Fin 16 → ℝ := fun i0 => (∑ lane : Fin 32, vmul lane i0)
    let vmul_1 : Fin 16 → ℝ := fun i0 => (vsum_4 i0 * (vsum_2 i0 : ℝ))
    let vsum_5 : Fin 1 → ℝ := fun i0 => (∑ lane : Fin 16, vmul_1 lane)
    let vmul_2 : Fin 1 → Int := fun i0 => (vclamp_min ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * 2)
    let vmul_3 : Fin 1 → Int := fun i0 => (vmul_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vclamp_min ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vreciprocal : Fin 1 → ℝ := fun i0 => (1 / ((vmul_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ : ℝ)))
    let vmul_4 : Fin 1 → ℝ := fun i0 => (vreciprocal ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (52/25))
    let vmul_5 : Fin 1 → ℝ := fun i0 => (vsum_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vmul_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_6 : Fin 1 → ℝ := fun i0 => (1 * vmul_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_7 : Fin 16 → ℝ := fun i0 => (vmul_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (vsum_2 i0 : ℝ))
    let vmul_8 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vmul_7 i1 * (if vbitwise_not i0 then 1 else 0))
    let vdiv_1 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vsigmoid i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_2 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vdiv_1 i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vneg : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (- (vmul_8 i0 i1))
    let vmul_9 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vneg i0 i1 * vdiv_2 i0 i1)
    let vdiv_3 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vmul_8 i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_6 : Fin 32 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 16, vmul_9 i0 lane)
    let vadd_1 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vdiv_3 i0 i1 + vsum_6 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsigmoid_backward : Fin 32 → Fin 16 → ℝ := fun i0 i1 => (vadd_1 i0 i1 * vsigmoid i0 i1 * (1 - vsigmoid i0 i1))
    let vto_copy_3 : Fin 32 → Fin 16 → ℝ := fun i0 i1 => vsigmoid_backward i0 i1
    vto_copy_3 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_masked_moe_aux_dx
