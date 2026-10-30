import bench.pretrain_kernels.tasks.local_fused_swiglu_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_swiglu_dx
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
    let input1 := xs.input1
    let vto_copy : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => input0 i0 ⟨(i1.val) % 1280, Nat.mod_lt _ (by decide)⟩
    let vsilu : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vto_copy i0 i1 * (1 / (1 + Real.exp (-(vto_copy i0 i1)))))
    let vto_copy_1 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => input0 i0 ⟨((i1.val + 640)) % 1280, Nat.mod_lt _ (by decide)⟩
    let vto_copy_3 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => input1 i0 i1
    let vmul_1 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vto_copy_3 i0 i1 * vsilu i0 i1)
    let vmul_2 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vto_copy_3 i0 i1 * vto_copy_1 i0 i1)
    let vto_copy_4 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => vmul_1 i0 i1
    let vsilu_backward : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vmul_2 i0 i1 * (1 / (1 + Real.exp (-(vto_copy i0 i1)))) * (1 + vto_copy i0 i1 * (1 - (1 / (1 + Real.exp (-(vto_copy i0 i1)))))))
    let vto_copy_5 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => vsilu_backward i0 i1
    let vcat : Fin 4096 → Fin 1280 → ℝ := fun i0 i1 => (if i1.val < 640 then vto_copy_5 i0 ⟨((i1.val - 0)) % 640, Nat.mod_lt _ (by decide)⟩ else vto_copy_4 i0 ⟨((i1.val - 640)) % 640, Nat.mod_lt _ (by decide)⟩)
    vcat i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_swiglu_dx
