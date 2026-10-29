import bench.pretrain_kernels.tasks.local_muon_syrk_gram.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
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
    let input0 := xs.input0
    let vbmm : Fin 3 → Fin 1544 → Fin 1544 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 520, input0 i0 i1 lane * input0 i0 i2 lane)
    vbmm i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
