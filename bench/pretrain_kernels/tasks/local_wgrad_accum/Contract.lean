import bench.pretrain_kernels.tasks.local_wgrad_accum.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_wgrad_accum
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
    let parameter0 := xs.parameter0
    let input0 := xs.input0
    let input1 := xs.input1
    let vmul : Fin 6144 → Fin 2048 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (16 : ℝ))
    let vmm : Fin 6144 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, input1 lane i0 * input0 lane i1)
    let vadd : Fin 6144 → Fin 2048 → ℝ := fun i0 i1 => (vmul i0 i1 + vmm i0 i1)
    vadd i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_wgrad_accum
