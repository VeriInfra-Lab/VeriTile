import bench.pretrain_kernels.tasks.local_fused_score_for_aux_loss.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss
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
    let vsigmoid : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (1 / (1 + Real.exp (-(input0 i0 i1))))
    let vsum_1 : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 128, vsigmoid i0 lane)
    let vadd : Fin 256 → Fin 1 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/100000000000000000000))
    let vdiv : Fin 256 → Fin 128 → ℝ := fun i0 i1 => (vsigmoid i0 i1 / vadd i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    vdiv i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_score_for_aux_loss
