import bench.pretrain_kernels.tasks.local_fused_unpermute_dmerging_probs.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
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
let vtopk_index : Fin 256 → Fin 6 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int);
let vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 6).foldl (fun value lane => if vtopk_index i0 ⟨(lane.val) % 6, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false));
let vto_copy_7 : Fin 32768 → Int := fun i0 => (if vscatter ⟨((i0.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then 1 else 0);
(∑ expert : Fin 32768, ((vto_copy_7 ⟨(expert.val) % 32768, Nat.mod_lt _ (by decide)⟩ + 0) / 1) * 1) = 1536)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let input0 := xs.input0
    let vtopk_value : Fin 256 → Fin 6 → ℝ := fun i0 i1 => parameter0 i0 ⟨((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val) % 128, Nat.mod_lt _ (by decide)⟩
    let vtopk_index : Fin 256 → Fin 6 → Int := fun i0 i1 => ((((List.finRange 128).mergeSort (fun left right => decide (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ > parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∨ (parameter0 i0 ⟨(left.val) % 128, Nat.mod_lt _ (by decide)⟩ = parameter0 i0 ⟨(right.val) % 128, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i1.val 0).val : Int)
    let vscatter : Fin 256 → Fin 128 → Bool := fun i0 i1 => ((List.finRange 6).foldl (fun value lane => if vtopk_index i0 ⟨(lane.val) % 6, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then Bool.true else value) (Bool.false))
    let vsort_value : Fin 32768 → Bool := fun i0 => vscatter ⟨(((((List.finRange 32768).mergeSort (fun left right => decide ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) > (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∨ ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) = (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∧ left.val ≤ right.val)))).getD i0.val 0).val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((((List.finRange 32768).mergeSort (fun left right => decide ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) > (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∨ ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) = (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∧ left.val ≤ right.val)))).getD i0.val 0).val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩
    let vsort_index : Fin 32768 → Int := fun i0 => ((((List.finRange 32768).mergeSort (fun left right => decide ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) > (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∨ ((if vscatter ⟨((left.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((left.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) = (if vscatter ⟨((right.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((right.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Nat) else 0) ∧ left.val ≤ right.val)))).getD i0.val 0).val : Int)
    let vremainder : Fin 1536 → Int := fun i0 => (vsort_index ⟨(i0.val) % 32768, Nat.mod_lt _ (by decide)⟩ % 256)
    let vindex_select : Fin 1536 → Fin 2048 → ℝ := fun i0 i1 => parameter1 ⟨(Int.toNat (vremainder i0)) % 256, Nat.mod_lt _ (by decide)⟩ i1
    let vto_copy_6 : Fin 1536 → Fin 2048 → ℝ := fun i0 i1 => vindex_select i0 i1
    let vto_copy_7 : Fin 32768 → Int := fun i0 => (if vscatter ⟨((i0.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val / 256) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ then 1 else 0)
    let vsort_1_value : Fin 32768 → Int := fun i0 => vto_copy_7 ⟨((((List.finRange 32768).mergeSort (fun left right => decide (vto_copy_7 ⟨(left.val) % 32768, Nat.mod_lt _ (by decide)⟩ > vto_copy_7 ⟨(right.val) % 32768, Nat.mod_lt _ (by decide)⟩ ∨ (vto_copy_7 ⟨(left.val) % 32768, Nat.mod_lt _ (by decide)⟩ = vto_copy_7 ⟨(right.val) % 32768, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i0.val 0).val) % 32768, Nat.mod_lt _ (by decide)⟩
    let vsort_1_index : Fin 32768 → Int := fun i0 => ((((List.finRange 32768).mergeSort (fun left right => decide (vto_copy_7 ⟨(left.val) % 32768, Nat.mod_lt _ (by decide)⟩ > vto_copy_7 ⟨(right.val) % 32768, Nat.mod_lt _ (by decide)⟩ ∨ (vto_copy_7 ⟨(left.val) % 32768, Nat.mod_lt _ (by decide)⟩ = vto_copy_7 ⟨(right.val) % 32768, Nat.mod_lt _ (by decide)⟩ ∧ left.val ≤ right.val)))).getD i0.val 0).val : Int)
    let vto_copy_8 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => input0 i0 i1
    let vto_copy_9 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => vto_copy_8 i0 i1
    let vgather : Fin 1536 → Fin 2048 → ℝ := fun i0 i1 => vto_copy_9 ⟨(Int.toNat (vremainder i0)) % 256, Nat.mod_lt _ (by decide)⟩ i1
    let vmul_1 : Fin 1536 → Fin 2048 → ℝ := fun i0 i1 => (vgather i0 i1 * vto_copy_6 i0 i1)
    let vsum_3 : Fin 1536 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vmul_1 i0 lane)
    let vindex_put : Fin 32768 → ℝ := fun i0 => (List.finRange 1536).foldl (fun acc lane => if vsort_1_index ⟨(lane.val) % 32768, Nat.mod_lt _ (by decide)⟩ = (i0.val : Int) then acc + vsum_3 lane ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ else acc) (0)
    let vto_copy_10 : Fin 256 → Fin 128 → ℝ := fun i0 i1 => vindex_put ⟨(((i1.val * 256) + i0.val)) % 32768, Nat.mod_lt _ (by decide)⟩
    vto_copy_10 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_unpermute_dmerging_probs
