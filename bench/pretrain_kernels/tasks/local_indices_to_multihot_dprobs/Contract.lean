import bench.pretrain_kernels.tasks.local_indices_to_multihot_dprobs.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_indices_to_multihot_dprobs
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
    let input1 := xs.input1
    let vne : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 ≠ (-1))
    let vlt : Fin 128 → Fin 4 → Bool := fun i0 i1 => decide (input0 i0 i1 < 16)
    let vbitwise_and : Fin 128 → Fin 4 → Bool := fun i0 i1 => (vne i0 i1 && vlt i0 i1)
    let vwhere : Fin 128 → Fin 4 → Int := fun i0 i1 => (if vbitwise_and i0 i1 then input0 i0 i1 else 16)
    let vslice_backward : Fin 128 → Fin 17 → ℝ := fun i0 i1 => (if (i1.val ≥ 0 ∧ i1.val < 16 ∧ (i1.val - 0) % 1 = 0) then input1 i0 ⟨((i1.val - 0)) % 16, Nat.mod_lt _ (by decide)⟩ else 0)
    let vgather : Fin 128 → Fin 4 → ℝ := fun i0 i1 => vslice_backward i0 ⟨(Int.toNat (vwhere i0 i1)) % 17, Nat.mod_lt _ (by decide)⟩
    vgather i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_indices_to_multihot_dprobs
