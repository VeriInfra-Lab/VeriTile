import bench.pretrain_kernels.tasks.local_mla_rope_fused.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mla_rope_fused
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
    let vto_copy : Fin 4 → ℝ := fun i0 => ((0 + (i0.val : Int) * 2) : ℝ)
    let vdiv : Fin 4 → ℝ := fun i0 => (vto_copy i0 / 8)
    let vpow_1 : Fin 4 → ℝ := fun i0 => Real.rpow (10000) (vdiv i0)
    let vreciprocal : Fin 4 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul : Fin 4 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vto_copy_4 : Fin 1 → Fin 1 → Fin 128 → ℝ := fun i0 i1 i2 => ((0 + (i2.val : Int) * 1) : ℝ)
    let vbmm : Fin 1 → Fin 4 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1, vmul ⟨(((i0.val * 4) + i1.val + lane.val)) % 4, Nat.mod_lt _ (by decide)⟩ * vto_copy_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + (lane.val * 128) + i2.val)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_2 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (if i2.val < 4 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 4) + (i2.val - 0)) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i1 else vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 4) + (i2.val - 4)) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ i1)
    let vcos_1 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (Real.cos (vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_5 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (vcos_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (5/4))
    let vsin_1 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (Real.sin (vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_6 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => (vsin_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (5/4))
    let vto_copy_5 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => vmul_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_6 : Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 => vmul_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmul_7 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨(((i3.val * 2) + 16)) % 24, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vmul_8 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨(((i3.val * 2) + 17)) % 24, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vsub : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_7 i0 i1 i2 i3 - vmul_8 i0 i1 i2 i3)
    let vmul_9 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨(((i3.val * 2) + 17)) % 24, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vmul_10 : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input0 i0 i1 i2 ⟨(((i3.val * 2) + 16)) % 24, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vadd : Fin 2 → Fin 12 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_9 i0 i1 i2 i3 + vmul_10 i0 i1 i2 i3)
    let vcat_3 : Fin 2 → Fin 12 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 4 then vsub i0 i1 i2 ⟨((i3.val - 0)) % 4, Nat.mod_lt _ (by decide)⟩ else vadd i0 i1 i2 ⟨((i3.val - 4)) % 4, Nat.mod_lt _ (by decide)⟩)
    let vmul_11 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input2 i0 i2 ⟨((i3.val * 2)) % 8, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vmul_12 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input2 i0 i2 ⟨(((i3.val * 2) + 1)) % 8, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vsub_1 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_11 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_12 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_13 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input2 i0 i2 ⟨(((i3.val * 2) + 1)) % 8, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vmul_14 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (input2 i0 i2 ⟨((i3.val * 2)) % 8, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 8, Nat.mod_lt _ (by decide)⟩)
    let vadd_1 : Fin 2 → Fin 1 → Fin 128 → Fin 4 → ℝ := fun i0 i1 i2 i3 => (vmul_13 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_14 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_4 : Fin 2 → Fin 1 → Fin 128 → Fin 8 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 4 then vsub_1 i0 i1 i2 ⟨((i3.val - 0)) % 4, Nat.mod_lt _ (by decide)⟩ else vadd_1 i0 i1 i2 ⟨((i3.val - 4)) % 4, Nat.mod_lt _ (by decide)⟩)
    let vcat_5 : Fin 2 → Fin 12 → Fin 128 → Fin 24 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 16 then input0 i0 i1 i2 ⟨((i3.val - 0)) % 24, Nat.mod_lt _ (by decide)⟩ else vcat_3 i0 i1 i2 ⟨((i3.val - 16)) % 8, Nat.mod_lt _ (by decide)⟩)
    let vcat_6 : Fin 2 → Fin 12 → Fin 128 → Fin 24 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 16 then input1 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vcat_4 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 16)) % 8, Nat.mod_lt _ (by decide)⟩)
    let vcat_7 : Fin 2 → Fin 12 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 24 then vcat_5 i0 i1 i2 ⟨((i3.val - 0)) % 24, Nat.mod_lt _ (by decide)⟩ else (if i3.val < 48 then vcat_6 i0 i1 i2 ⟨((i3.val - 24)) % 24, Nat.mod_lt _ (by decide)⟩ else input1 i0 i1 i2 ⟨(((i3.val - 48) + 16)) % 32, Nat.mod_lt _ (by decide)⟩))
    vcat_7 i0 i1 i2 i3

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mla_rope_fused
