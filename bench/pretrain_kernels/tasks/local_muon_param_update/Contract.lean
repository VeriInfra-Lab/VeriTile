import bench.pretrain_kernels.tasks.local_muon_param_update.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_muon_param_update
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
  fun xs (i0,i1,i2) =>
    let parameter0 := xs.parameter0
    let input0 := xs.input0
    let vmul : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (parameter0 i0 i1 i2 * (1/10))
    let vto_copy : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => vmul i0 i1 i2
    let vmul_1 : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (input0 i0 i1 i2 * (9999/10000))
    let vto_copy_1 : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => vto_copy i0 i1 i2
    let vaddcmul : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (vmul_1 i0 i1 i2 + 1 * (-37/100000) * vto_copy_1 i0 i1 i2)
    vaddcmul i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_muon_param_update
