import bench.pretrain_kernels.tasks.local_fused_hc_post_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_post_fwd
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
  fun xs (i0,i1,i2,i3) =>
    let input0 := xs.input0
    let input1 := xs.input1
    let input2 := xs.input2
    let vto_copy : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => input0 i0 i1 i2
    let vto_copy_1 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => input1 i0 i1 i2 i3
    let vmul : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (input2 i0 i1 i2 * vto_copy i0 i1 i3)
    let vadd : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (vmul i0 i1 i2 i3 + vto_copy_1 i0 i1 i2 i3)
    let vto_copy_2 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => vadd i0 i1 i2 i3
    vto_copy_2 i0 i1 i2 i3

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_post_fwd
