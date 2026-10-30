import bench.pretrain_kernels.tasks.local_deepgemm_linear_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_linear_fwd
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
    let vmul : Fin 2048 → Fin 2048 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/50))
    let vmm : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, input0 ⟨(((((i0.val * 2048) + lane.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vmul i1 lane)
    vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_linear_fwd
