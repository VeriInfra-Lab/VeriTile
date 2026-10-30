import bench.pretrain_kernels.tasks.local_indices_to_multihot_map.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_indices_to_multihot_map
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
    let vne : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 ≠ (-1))
    let vlt : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 < 16)
    let vbitwise_and : Fin 128 → Fin 4 → Bool := fun i0 i1 => (vne i0 i1 && vlt i0 i1)
    let vwhere : Fin 128 → Fin 4 → Int := fun i0 i1 => (if vbitwise_and i0 i1 then input0 i0 i1 else 16)
    let vscatter : Fin 128 → Fin 17 → Int := fun i0 i1 => ((List.finRange 4).foldl (fun value lane => if vwhere i0 ⟨(lane.val) % 4, Nat.mod_lt _ (by decide)⟩ = (i1.val : Int) then 1 else value) (0))
    let vto_copy_1 : Fin 128 → Fin 16 → Bool := fun i0 i1 => decide (vscatter i0 ⟨(i1.val) % 17, Nat.mod_lt _ (by decide)⟩ ≠ 0)
    let vto_copy_2 : Fin 128 → Fin 16 → ℝ := fun i0 i1 => (if vto_copy_1 i0 i1 then 1 else 0)
    vto_copy_2 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_indices_to_multihot_map
