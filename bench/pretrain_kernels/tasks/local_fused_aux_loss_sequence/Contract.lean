import bench.pretrain_kernels.tasks.local_fused_aux_loss_sequence.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_aux_loss_sequence
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (0 < xs.configurationSmCount 0)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs i0 =>
    let input0 := xs.input0
    let input1 := xs.input1
    let configurationSmCount := xs.configurationSmCount
    let vsum_1 : Fin 384 → ℝ := fun i0 => (∑ lane : Fin 257, input0 lane i0)
    let vmul : Fin 384 → ℝ := fun i0 => (vsum_1 i0 * input1 i0)
    let vsum_2 : Fin 1 → ℝ := fun i0 => (∑ lane : Fin 384, vmul lane)
    let vmul_1 : Fin 1 → ℝ := fun i0 => (vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (1614962628755419/500000000000000000000))
    vmul_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_aux_loss_sequence
