import bench.pretrain_kernels.tasks.local_mla_fa4_te_rope_cp_absorb_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
namespace Contract
open scoped BigOperators Classical

def Allowed (xs : IO.Inputs) : Prop :=
  (∀ (j0 : Fin 128), xs.constantee33028f0f j0 = ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 48, 49, 50, 51, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 60, 61, 62, 63, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 52, 53, 54, 55, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 56, 57, 58, 59, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111] : List Int).getD (j0.val) 0)

def output : IO.Inputs → IO.OutputIndex → ℝ :=
  fun xs (i0,i1,i2) =>
    let parameter0 := xs.parameter0
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let parameter3 := xs.parameter3
    let parameter4 := xs.parameter4
    let parameter5 := xs.parameter5
    let parameter6 := xs.parameter6
    let input0 := xs.input0
    let input1 := xs.input1
    let constantee33028f0f := xs.constantee33028f0f
    let vmul : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_1 : Fin 256 → ℝ := fun i0 => (parameter1 i0 * (1/10))
    let vadd : Fin 256 → ℝ := fun i0 => (vmul_1 i0 + 1)
    let vmul_2 : Fin 1536 → Fin 256 → ℝ := fun i0 i1 => (parameter2 i0 i1 * (1/16))
    let vmul_3 : Fin 576 → Fin 512 → ℝ := fun i0 i1 => (parameter3 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_4 : Fin 512 → ℝ := fun i0 => (parameter4 i0 * (1/10))
    let vadd_1 : Fin 512 → ℝ := fun i0 => (vmul_4 i0 + 1)
    let vmul_5 : Fin 2048 → Fin 512 → ℝ := fun i0 i1 => (parameter5 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => (parameter6 i0 i1 * (1/32))
    let vto_copy : Fin 256 → Fin 512 → ℝ := fun i0 i1 => vmul i0 i1
    let vto_copy_1 : Fin 256 → ℝ := fun i0 => vadd i0
    let vto_copy_2 : Fin 1536 → Fin 256 → ℝ := fun i0 i1 => vmul_2 i0 i1
    let vto_copy_3 : Fin 576 → Fin 512 → ℝ := fun i0 i1 => vmul_3 i0 i1
    let vto_copy_4 : Fin 512 → ℝ := fun i0 => vadd_1 i0
    let vto_copy_5 : Fin 2048 → Fin 512 → ℝ := fun i0 i1 => vmul_5 i0 i1
    let vto_copy_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => vmul_6 i0 i1
    let vcat : Fin 128 → Int := fun i0 => (if i0.val < 48 then (0 + ((i0.val - 0) : Int) * 1) else (if i0.val < 64 then (0 + ((i0.val - 48) : Int) * 1) else (0 + ((i0.val - 64) : Int) * 1)))
    let vadd_2 : Fin 1 → Fin 128 → Int := fun i0 i1 => (vcat i1 + 2)
    let vto_copy_14 : Fin 32 → ℝ := fun i0 => ((0 + (i0.val : Int) * 2) : ℝ)
    let vdiv : Fin 32 → ℝ := fun i0 => (vto_copy_14 i0 / 64)
    let vpow_1 : Fin 32 → ℝ := fun i0 => Real.rpow ((10000 : ℝ)) (vdiv i0)
    let vreciprocal : Fin 32 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul_7 : Fin 32 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vto_copy_16 : Fin 1 → Fin 1 → Fin 128 → ℝ := fun i0 i1 i2 => (vadd_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 : ℝ)
    let vbmm : Fin 1 → Fin 32 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1, vmul_7 ⟨(((i0.val * 32) + i1.val + lane.val)) % 32, Nat.mod_lt _ (by decide)⟩ * vto_copy_16 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + (lane.val * 128) + i2.val)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_1 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (if i2.val < 32 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 32) + (i2.val - 0)) % 32)) % 32, Nat.mod_lt _ (by decide)⟩ i1 else vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 32) + (i2.val - 32)) % 32)) % 32, Nat.mod_lt _ (by decide)⟩ i1)
    let vcos : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (Real.cos (vcat_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_9 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (vcos ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vsin : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (Real.sin (vcat_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_10 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (vsin ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vto_copy_17 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vmul_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_18 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vmul_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vsub : Fin 1 → Fin 1 → Int := fun i0 i1 => (vadd_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 128, Nat.mod_lt _ (by decide)⟩ - 1)
    let vcat_3 : Fin 1 → Fin 129 → Int := fun i0 i1 => (if i1.val < 1 then vsub i0 ⟨((i1.val - 0)) % 1, Nat.mod_lt _ (by decide)⟩ else vadd_2 i0 ⟨((i1.val - 1)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vsub_1 : Fin 1 → Fin 128 → Int := fun i0 i1 => (vcat_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((i1.val + 1)) % 129, Nat.mod_lt _ (by decide)⟩ - vcat_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(i1.val) % 129, Nat.mod_lt _ (by decide)⟩)
    let vne : Fin 1 → Fin 128 → Bool := fun i0 i1 => decide (vsub_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ≠ 1)
    let vcumsum_1 : Fin 1 → Fin 128 → Int := fun i0 i1 => (∑ lane : Fin 128, if lane.val ≤ i1.val then (if vne i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Int) else 0) else 0)
    let veq : Fin 1 → Fin 1 → Fin 128 → Fin 128 → Bool := fun i0 i1 i2 i3 => decide (vcumsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 = vcumsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3)
    let vtril : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≤ i0.val + 0 then Bool.true else Bool.false)
    let vbitwise_and : Fin 1 → Fin 1 → Fin 128 → Fin 128 → Bool := fun i0 i1 i2 i3 => (veq ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 && vtril i2 i3)
    let vwhere : Fin 1 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vbitwise_and ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 then 0 else (-338953138925153550000000000000000000000 : ℝ))
    let vmm : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy i1 lane)
    let vto_copy_20 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm i1 i2
    let vpow_3 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_3 i0 i1 lane) / 256)
    let vadd_3 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_13 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_14 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_13 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_21 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_1 : Fin 128 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 128 → Fin 576 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_3 i1 lane)
    let vto_copy_22 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmm_2 i1 ⟨(i2.val) % 576, Nat.mod_lt _ (by decide)⟩
    let vpow_4 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => ((vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 512, vpow_4 i0 i1 lane) / 512)
    let vadd_4 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_15 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_16 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_23 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmul_16 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_3 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 512, vto_copy_23 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_5 i1 lane)
    let vmul_17 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_18 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub_9 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 - vmul_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vmul_19 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_20 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_5 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vcat_5 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_9 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_5 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vmul_21 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 512)) % 576, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_22 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 513)) % 576, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub_10 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_23 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 513)) % 576, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_24 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 512)) % 576, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_6 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_23 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_24 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_6 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_10 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_6 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcat_7 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_1 ⟨(((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 1536) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 1536)) % 1536, Nat.mod_lt _ (by decide)⟩ else vcat_5 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_8 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_3 ⟨(((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 2048) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ else vcat_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vbmm_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vcat_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i1 lane * vcat_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i2 lane)
    let vmul_25 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm_2 i1 i2 i3 * (3608439182435161/50000000000000000))
    let vadd_7 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul_25 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vwhere ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy_24 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vsoftmax : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_24 i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy_24 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_25 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmm_5 : Fin 128 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_6 lane i1)
    let vbmm_4 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_25 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmm_5 lane ⟨(((i0.val * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_5 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_5 i1 ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vmm_3 i2 ⟨(((i0.val * 256) + lane.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_26 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vbmm_5 i1 i2 i3
    let vsoftmax_backward_data : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_26 i0 i1 i2 i3 - (∑ lane : Fin 128, vto_copy_26 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2 i3
    let vto_copy_27 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax_backward_data ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmul_26 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_27 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 * (3608439182435161/50000000000000000))
    let vbmm_6 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vcat_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmul_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i2)
    let vbmm_7 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmul_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i1 lane * vcat_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i2)
    let vsum_1 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 8, vbmm_6 ⟨(((i0.val * 8) + lane.val)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ i2)
    let vmul_27 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_28 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vneg : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩))
    let vmul_29 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_8 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_28 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_29 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_30 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_9 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_27 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_30 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_31 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_7 i1 i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_32 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_7 i1 i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vneg_1 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vbmm_7 i1 i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩))
    let vmul_33 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_10 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_32 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_33 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vmul_34 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_7 i1 i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_11 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_31 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_34 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vslice_backward : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_8 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_9 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_12 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vslice_backward_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vslice_backward_2 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_10 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_3 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_11 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_13 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vslice_backward_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vcat_9 : Fin 1 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_6 ⟨(((((i0.val * 196608) + (i1.val * 24576) + ((i3.val - 0) * 128) + i2.val) / 24576) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ i2 else vbmm_4 ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 16384) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 128) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_10 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_7 ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 24576) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 192) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ else vadd_13 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_6 : Fin 128 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vcat_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 2048) + lane.val) / 256) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i0 ⟨((((i0.val * 2048) + lane.val) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 lane i1)
    let vto_copy_28 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmm_6 i1 i2
    let vmul_35 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_28 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_4 i2)
    let vmul_36 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_35 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vmul_37 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_35 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_2 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 512, vmul_36 i0 i1 lane)
    let vpow_5 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_38 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_39 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_38 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_2 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_39 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 512)
    let vpow_6 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => ((vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_40 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vpow_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (2 : ℝ))
    let vmul_41 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vdiv_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vmul_40 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vadd_14 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_37 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 + vmul_41 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_29 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vadd_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vcat_11 : Fin 1 → Fin 128 → Fin 576 → ℝ := fun i0 i1 i2 => (if i2.val < 512 then vto_copy_29 i0 i1 ⟨((i2.val - 0)) % 512, Nat.mod_lt _ (by decide)⟩ else vadd_12 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 8192) + (i1.val * 64) + (i2.val - 512)) / 64) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 8192) + (i1.val * 64) + (i2.val - 512)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_7 : Fin 128 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 576, vcat_11 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_3 lane i1)
    let vmm_8 : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 1536, vcat_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 1536) + lane.val) / 192) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i0 ⟨((((i0.val * 1536) + lane.val) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_2 lane i1)
    let vto_copy_30 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_8 i1 i2
    let vmul_42 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_30 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_1 i2)
    let vmul_43 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_42 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vmul_44 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_42 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_3 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vmul_43 i0 i1 lane)
    let vpow_7 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_45 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_46 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_45 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_3 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_46 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 256)
    let vpow_8 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_47 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vpow_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (2 : ℝ))
    let vmul_48 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vdiv_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vmul_47 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vadd_15 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_44 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 + vmul_48 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_31 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vadd_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_9 : Fin 128 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_31 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy lane i1)
    let vadd_16 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmm_7 i1 i2 + vmm_9 i1 i2)
    vadd_16 i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_cp_absorb_dx
