import bench.pretrain_kernels.tasks.local_fused_permute_padded.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_permute_padded
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
    let input0 := xs.input0
    let vtopk_value : Fin 256 → Fin 1 → ℝ := fun i0 i1 => parameter0 i0 ⟨((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 128, Nat.mod_lt _ (by decide)⟩
    let vtopk_index : Fin 256 → Fin 1 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 1).foldl (fun value lane => if vtopk_index i0 ⟨(lane.val) % 1, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false))
    let vto_copy_5 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => input0 i0 i1
    let vto_copy_6 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if vscatter i0 i1 then 1 else 0)
    let vsum_3 : Fin 128 → Int := fun i0 => (∑ lane : Fin 256, vto_copy_6 lane i0)
    let vadd_2 : Fin 128 → Int := fun i0 => (vsum_3 i0 + 16)
    let vsub_3 : Fin 128 → Int := fun i0 => (vadd_2 i0 - 1)
    let vfloor_divide_1 : Fin 128 → Int := fun i0 => (vsub_3 i0 / 16)
    let vmul_1 : Fin 128 → Int := fun i0 => (vfloor_divide_1 i0 * 16)
    let vcumsum_2 : Fin 128 → Int := fun i0 => (∑ lane : Fin 128, if lane.val ≤ i0.val then vmul_1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    let vsub_4 : Fin 128 → Int := fun i0 => (vcumsum_2 i0 - vmul_1 i0)
    let vto_copy_7 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if vscatter i0 i1 then 1 else 0)
    let vcumsum_3 : Fin 256 → Fin 128 → Int := fun i0 i1 => (∑ lane : Fin 256, if lane.val ≤ i0.val then vto_copy_7 ⟨(lane.val) % 256, Nat.mod_lt _ (by decide)⟩ i1 else 0)
    let vsub_5 : Fin 256 → Fin 128 → Int := fun i0 i1 => (vcumsum_3 i0 i1 - 1)
    let vadd_3 : Fin 256 → Fin 128 → Int := fun i0 i1 => (vsub_5 i0 i1 + vsub_4 i1)
    let vwhere_4 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if vscatter i0 i1 then vadd_3 i0 i1 else (-1))
    let veq : Fin 1856 → Fin 32768 → Bool := fun i0 i1 => decide ((0 + (i0.val : Int) * 1) = vwhere_4 ⟨(((i1.val / 128) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((i1.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vwhere_5 : Fin 1856 → Fin 32768 → Int := fun i0 i1 => (if veq i0 i1 then (0 + (((i1.val / 128) % 256) : Int) * 1) else 0)
    let vsum_4 : Fin 1856 → Int := fun i0 => (∑ lane : Fin 32768, vwhere_5 i0 lane)
    let vany_2 : Fin 1856 → Bool := fun i0 => decide (∃ lane : Fin 32768, veq i0 ⟨(lane.val) % 32768, Nat.mod_lt _ (by decide)⟩)
    let vindex : Fin 1856 → Fin 2048 → ℝ := fun i0 i1 => vto_copy_5 ⟨(Int.toNat (vsum_4 i0)) % 256, Nat.mod_lt _ (by decide)⟩ i1
    let vwhere_6 : Fin 1856 → Fin 2048 → ℝ := fun i0 i1 => (if vany_2 i0 then vindex i0 i1 else (0 : ℝ))
    vwhere_6 i0 i1))

end
end VeriTile.Bench.PretrainKernels.local_fused_permute_padded
