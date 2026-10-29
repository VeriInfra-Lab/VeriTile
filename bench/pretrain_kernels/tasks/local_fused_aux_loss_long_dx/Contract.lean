import bench.pretrain_kernels.tasks.local_fused_aux_loss_long_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_dx
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
    let vmul_2 : Fin 1 → ℝ := fun i0 => (input1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (6357828776041667/500000000000000000000000))
    let vmul_3 : Fin 128 → ℝ := fun i0 => (vmul_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * input0 i0)
    vmul_3 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_long_dx
