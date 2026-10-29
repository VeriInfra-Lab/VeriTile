import bench.pretrain_kernels.tasks.local_situ_glu_b30x30_transition_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_transition_dx
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
    let vto_copy : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨(i2.val) % 12288, Nat.mod_lt _ (by decide)⟩
    let vto_copy_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨((i2.val + 6144)) % 12288, Nat.mod_lt _ (by decide)⟩
    let vdiv : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 / (30 : ℝ))
    let vtanh : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (Real.tanh (vdiv i0 i1 i2))
    let vmul : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh i0 i1 i2 * (30 : ℝ))
    let vsigmoid : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (1 / (1 + Real.exp (-(vto_copy i0 i1 i2))))
    let vmul_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul i0 i1 i2 * vsigmoid i0 i1 i2)
    let vdiv_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy_1 i0 i1 i2 / (30 : ℝ))
    let vtanh_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (Real.tanh (vdiv_1 i0 i1 i2))
    let vmul_2 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh_1 i0 i1 i2 * (30 : ℝ))
    let vto_copy_3 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => input1 i0 i1 i2
    let vmul_4 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy_3 i0 i1 i2 * vmul_1 i0 i1 i2)
    let vmul_5 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy_3 i0 i1 i2 * vmul_2 i0 i1 i2)
    let vmul_6 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_4 i0 i1 i2 * (30 : ℝ))
    let vtanh_backward : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_6 i0 i1 i2 * (1 - vtanh_1 i0 i1 i2 * vtanh_1 i0 i1 i2))
    let vdiv_2 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh_backward i0 i1 i2 / (30 : ℝ))
    let vmul_7 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_5 i0 i1 i2 * vmul i0 i1 i2)
    let vmul_8 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_5 i0 i1 i2 * vsigmoid i0 i1 i2)
    let vsigmoid_backward : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_7 i0 i1 i2 * vsigmoid i0 i1 i2 * (1 - vsigmoid i0 i1 i2))
    let vmul_9 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_8 i0 i1 i2 * (30 : ℝ))
    let vtanh_backward_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_9 i0 i1 i2 * (1 - vtanh i0 i1 i2 * vtanh i0 i1 i2))
    let vdiv_3 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh_backward_1 i0 i1 i2 / (30 : ℝ))
    let vadd : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vsigmoid_backward i0 i1 i2 + vdiv_3 i0 i1 i2)
    let vto_copy_4 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vdiv_2 i0 i1 i2
    let vto_copy_5 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vadd i0 i1 i2
    let vcat : Fin 2 → Fin 128 → Fin 12288 → ℝ := fun i0 i1 i2 => (if i2.val < 6144 then vto_copy_5 i0 i1 ⟨((i2.val - 0)) % 6144, Nat.mod_lt _ (by decide)⟩ else vto_copy_4 i0 i1 ⟨((i2.val - 6144)) % 6144, Nat.mod_lt _ (by decide)⟩)
    vcat i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_situ_glu_b30x30_transition_dx
