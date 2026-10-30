import bench.pretrain_kernels.tasks.local_chunked_lm_loss_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (∀ (j0 : Fin 320), 0 ≤ xs.parameter1 j0 ∧ xs.parameter1 j0 < 1024)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs i0 =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let input0 := xs.input0
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
    let vlog : Fin 320 → ℝ := fun i0 => (Real.log (vsum_1 i0))
    let vgather : Fin 320 → Fin 1 → ℝ := fun i0 i1 => vto_copy_1 i0 ⟨(Int.toNat (parameter1 ⟨((i0.val + i1.val)) % 320, Nat.mod_lt _ (by decide)⟩)) % 1024, Nat.mod_lt _ (by decide)⟩
    let vsub_1 : Fin 320 → ℝ := fun i0 => (vgather i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ - vmax_1_value i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsub_2 : Fin 320 → ℝ := fun i0 => (vsub_1 i0 - vlog i0)
    let vneg : Fin 320 → ℝ := fun i0 => (- (vsub_2 i0))
    let vmul_1 : Fin 320 → ℝ := fun i0 => (vneg i0 * vto_copy i0)
    let vsum_2 : Fin 1 → ℝ := fun i0 => (∑ lane : Fin 320, vmul_1 lane)
    vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩

end Contract
end
end VeriTile.Bench.PretrainKernels.local_chunked_lm_loss_fwd
