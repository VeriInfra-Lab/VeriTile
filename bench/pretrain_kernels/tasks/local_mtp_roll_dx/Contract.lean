import bench.pretrain_kernels.tasks.local_mtp_roll_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mtp_roll_dx
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
    let vslice_backward : Fin 2 → Fin 128 → ℝ := fun i0 i1 => (if (i1.val ≥ 1 ∧ i1.val < 128 ∧ (i1.val - 1) % 1 = 0) then input0 i0 ⟨((i1.val - 1)) % 128, Nat.mod_lt _ (by decide)⟩ else 0)
    vslice_backward i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mtp_roll_dx
