import bench.pretrain_kernels.tasks.local_mtp_row_boundaries_closed_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mtp_row_boundaries_closed_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  ((∑ j : Fin 127, if xs.input0 ⟨j.val, Nat.lt_trans j.isLt (by decide)⟩ = 2 then (1 : Nat) else 0) = 2 ∧ xs.input0 ⟨127, by decide⟩ = 2)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs i0 =>
    let input0 := xs.input0
    let veq : Fin 127 → Bool := fun i0 => decide (input0 ⟨(i0.val) % 128, Nat.mod_lt _ (by decide)⟩ = 2)
    let vwhere : Fin 127 → Int := fun i0 => (if veq i0 then (1 + (i0.val : Int) * 1) else 129)
    let vsort_value : Fin 127 → Int := fun i0 => vwhere ⟨((((List.finRange 127).mergeSort (fun left right => decide (vwhere ⟨(left.val) % 127, Nat.mod_lt _ (by decide)⟩ < vwhere ⟨(right.val) % 127, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere ⟨(left.val) % 127, Nat.mod_lt _ (by decide)⟩ = vwhere ⟨(right.val) % 127, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i0.val 0).val) % 127, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 127 → Int := fun i0 => ((((List.finRange 127).mergeSort (fun left right => decide (vwhere ⟨(left.val) % 127, Nat.mod_lt _ (by decide)⟩ < vwhere ⟨(right.val) % 127, Nat.mod_lt _ (by decide)⟩ ∨ (vwhere ⟨(left.val) % 127, Nat.mod_lt _ (by decide)⟩ = vwhere ⟨(right.val) % 127, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i0.val 0).val : Int)
    let vcat : Fin 4 → Int := fun i0 => (if i0.val < 1 then 0 else (if i0.val < 3 then vsort_value ⟨((i0.val - 1)) % 127, Nat.mod_lt _ (by decide)⟩ else 128))
    let vto_copy : Fin 4 → ℝ := fun i0 => (vcat i0 : ℝ)
    let vto_copy_1 : Fin 4 → ℝ := fun i0 => vto_copy i0
    vto_copy_1 i0

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mtp_row_boundaries_closed_fwd
