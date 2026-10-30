import bench.pretrain_kernels.tasks.local_deepgemm_grouped_wgrad.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_wgrad
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
    let input1 := xs.input1
    let vmm_2 : Fin 640 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 128, input1 ⟨((lane.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ i0 * input0 ⟨((lane.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ i1)
    let vselect_backward : Fin 2 → Fin 640 → Fin 2048 → ℝ := fun i0 i1 i2 => (if (i0.val = 1) then vmm_2 i1 i2 else 0)
    let vmm_3 : Fin 640 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 128, input1 ⟨(lane.val) % 256, Nat.mod_lt _ (by decide)⟩ i0 * input0 ⟨(lane.val) % 256, Nat.mod_lt _ (by decide)⟩ i1)
    let vselect_backward_1 : Fin 2 → Fin 640 → Fin 2048 → ℝ := fun i0 i1 i2 => (if (i0.val = 0) then vmm_3 i1 i2 else 0)
    let vadd : Fin 2 → Fin 640 → Fin 2048 → ℝ := fun i0 i1 i2 => (vselect_backward i0 i1 i2 + vselect_backward_1 i0 i1 i2)
    vadd i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_wgrad
