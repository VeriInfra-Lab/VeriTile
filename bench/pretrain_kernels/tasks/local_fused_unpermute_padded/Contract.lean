import bench.pretrain_kernels.tasks.local_fused_unpermute_padded.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_padded
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  True ∧
    (let parameter0 := xs.parameter0;
let candidate_vtopk_index : Fin 256 → Fin 6 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int);
let candidate_vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 6).foldl (fun value lane => if candidate_vtopk_index i0 ⟨(lane.val) % 6, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false));
let candidate_vto_copy_2 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if candidate_vscatter i0 i1 then 1 else 0);
let candidate_vto_copy_3 : Fin 256 → Fin 128 → Bool := fun i0 i1 => decide (candidate_vto_copy_2 i0 i1 ≠ 0);
let candidate_vto_copy_4 : Fin 256 → Fin 128 → Int := fun i0 i1 => (if candidate_vto_copy_3 i0 i1 then 1 else 0);
let candidate_vsum_2 : Fin 128 → Int := fun i0 => (∑ lane : Fin 256, candidate_vto_copy_4 lane i0);
(∑ expert : Fin 128, ((candidate_vsum_2 ⟨(expert.val) % 128, Nat.mod_lt _ (by decide)⟩ + 15) / 16) * 16) = 2176)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let input0 := xs.input0
    let parameter0 := xs.parameter0
    let vto_copy_1 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => input0 i0 i1
    vto_copy_1 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_padded
