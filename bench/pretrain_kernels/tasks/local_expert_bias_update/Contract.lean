import bench.pretrain_kernels.tasks.local_expert_bias_update.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_expert_bias_update
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
    let vsum_1 : Fin 4 → Fin 1 → ℝ := fun i0 i1 => (∑ lane : Fin 128, input0 i0 lane)
    let vmul : Fin 4 → Fin 128 → ℝ := fun i0 i1 => (input0 i0 i1 * 128)
    let vsub : Fin 4 → Fin 128 → ℝ := fun i0 i1 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ - vmul i0 i1)
    let vsign : Fin 4 → Fin 128 → ℝ := fun i0 i1 => (if vsub i0 i1 > 0 then 1 else if vsub i0 i1 < 0 then -1 else 0)
    let vmul_1 : Fin 4 → Fin 128 → ℝ := fun i0 i1 => (vsign i0 i1 * (1/1000))
    let vadd : Fin 4 → Fin 128 → ℝ := fun i0 i1 => (((-1/20) + (((i0.val * 128) + i1.val) : ℝ) * (3913894324853229/20000000000000000000)) + vmul_1 i0 i1)
    vadd i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_expert_bias_update
