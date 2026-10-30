import bench.pretrain_kernels.tasks.local_permute.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_permute
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
    let parameter0 := xs.parameter0
    let vto_copy_1 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => input0 i0 i1
    vto_copy_1 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_permute
