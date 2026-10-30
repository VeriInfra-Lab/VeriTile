import bench.pretrain_kernels.tasks.local_fused_hc_head_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_head_dx
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
    let input3 := xs.input3
    let input4 := xs.input4
    let vto_copy : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨(((((i0.val * 33554432) + (i1.val * 8192) + i2.val) / 2048) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 33554432) + (i1.val * 8192) + i2.val) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩
    let vpow_1 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => ((vto_copy i0 i1 i2) * (vto_copy i0 i1 i2))
    let vmean : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 8192, vpow_1 i0 i1 lane) / 8192)
    let vadd : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmm : Fin 8192 → Fin 4 → ℝ := fun i0 i1 => (∑ lane : Fin 8192, vto_copy ⟨(((((i0.val * 8192) + lane.val) / 33554432) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4096)) % 4096, Nat.mod_lt _ (by decide)⟩ lane * input1 i1 lane)
    let vmul : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmm ⟨(((i0.val * 4096) + i1.val)) % 8192, Nat.mod_lt _ (by decide)⟩ i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_1 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul i0 i1 i2 * input2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vadd_1 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul_1 i0 i1 i2 + input3 i2)
    let vsigmoid : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (1 / (1 + Real.exp (-(vadd_1 i0 i1 i2))))
    let vadd_2 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vsigmoid i0 i1 i2 + (1/1000000))
    let vto_copy_2 : Fin 2 → Fin 4096 → Fin 2048 → ℝ := fun i0 i1 i2 => input4 i0 i1 i2
    let vmul_3 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (vto_copy_2 i0 i1 i3 * vadd_2 i0 i1 i2)
    let vmul_4 : Fin 2 → Fin 4096 → Fin 4 → Fin 2048 → ℝ := fun i0 i1 i2 i3 => (vto_copy_2 i0 i1 i3 * vto_copy i0 i1 ⟨(((i2.val * 2048) + i3.val)) % 8192, Nat.mod_lt _ (by decide)⟩)
    let vsum_2 : Fin 2 → Fin 4096 → Fin 4 → Fin 1 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 2048, vmul_4 i0 i1 i2 lane)
    let vsigmoid_backward : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vsum_2 i0 i1 i2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vsigmoid i0 i1 i2 * (1 - vsigmoid i0 i1 i2))
    let vmul_5 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vsigmoid_backward i0 i1 i2 * input2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_6 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul_5 i0 i1 i2 * vmm ⟨(((i0.val * 4096) + i1.val)) % 8192, Nat.mod_lt _ (by decide)⟩ i2)
    let vmul_7 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul_5 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_3 : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 4, vmul_6 i0 i1 lane)
    let vmm_1 : Fin 8192 → Fin 8192 → ℝ := fun i0 i1 => (∑ lane : Fin 4, vmul_7 ⟨(((((i0.val * 4) + lane.val) / 16384) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4096)) % 4096, Nat.mod_lt _ (by decide)⟩ lane * input1 lane i1)
    let vadd_3 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => (vmul_3 i0 i1 ⟨(((((i0.val * 33554432) + (i1.val * 8192) + i2.val) / 2048) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 33554432) + (i1.val * 8192) + i2.val) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ + vmm_1 ⟨(((i0.val * 4096) + i1.val)) % 8192, Nat.mod_lt _ (by decide)⟩ i2)
    let vpow_2 : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_8 : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_9 : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_8 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => (vmul_9 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 8192)
    let vpow_3 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => ((vto_copy i0 i1 i2))
    let vmul_10 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => (vpow_3 i0 i1 i2 * (2 : ℝ))
    let vmul_11 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => (vdiv i0 i1 i2 * vmul_10 i0 i1 i2)
    let vadd_4 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => (vadd_3 i0 i1 i2 + vmul_11 i0 i1 i2)
    let vto_copy_3 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => vadd_4 i0 i1 i2
    vto_copy_3 i0 i1 ⟨(((i2.val * 2048) + i3.val)) % 8192, Nat.mod_lt _ (by decide)⟩

end Contract
end
end VeriTile.Bench.PretrainKernels.local_fused_hc_head_dx
