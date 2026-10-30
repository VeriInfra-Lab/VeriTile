import bench.pretrain_kernels.tasks.local_deepgemm_grouped_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_dx
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
    let vmul : Fin 2 → Fin 640 → Fin 2048 → ℝ := fun i0 i1 i2 => (parameter0 i0 i1 i2 * (1/50))
    let vmm_2 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 640, input0 ⟨((i0.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ lane * vmul ⟨(1) % 2, Nat.mod_lt _ (by decide)⟩ lane i1)
    let vslice_backward : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (if (i0.val ≥ 128 ∧ i0.val < 256 ∧ (i0.val - 128) % 1 = 0) then vmm_2 ⟨((i0.val - 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1 else 0)
    let vmm_3 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 640, input0 ⟨(i0.val) % 256, Nat.mod_lt _ (by decide)⟩ lane * vmul ⟨(0) % 2, Nat.mod_lt _ (by decide)⟩ lane i1)
    let vslice_backward_1 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (if (i0.val ≥ 0 ∧ i0.val < 128 ∧ (i0.val - 0) % 1 = 0) then vmm_3 ⟨((i0.val - 0)) % 128, Nat.mod_lt _ (by decide)⟩ i1 else 0)
    let vadd : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (vslice_backward i0 i1 + vslice_backward_1 i0 i1)
    vadd i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_deepgemm_grouped_dx
