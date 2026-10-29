import bench.pretrain_kernels.tasks.local_triton_router_matmul_default.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_triton_router_matmul_default
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
    let vmul : Fin 65 → Fin 257 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/50))
    let vmm : Fin 65 → Fin 65 → ℝ := fun i0 i1 => (∑ lane : Fin 257, input0 i0 lane * vmul i1 lane)
    vmm i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_triton_router_matmul_default
