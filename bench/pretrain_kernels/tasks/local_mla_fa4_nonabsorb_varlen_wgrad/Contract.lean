import bench.pretrain_kernels.tasks.local_mla_fa4_nonabsorb_varlen_wgrad.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_varlen_wgrad
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
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let parameter3 := xs.parameter3
    let parameter4 := xs.parameter4
    let parameter5 := xs.parameter5
    let parameter6 := xs.parameter6
    let input0 := xs.input0
    let input1 := xs.input1
    let vmul : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_1 : Fin 256 → ℝ := fun i0 => (parameter1 i0 * (1/10))
    let vadd : Fin 256 → ℝ := fun i0 => (vmul_1 i0 + 1)
    let vmul_2 : Fin 1536 → Fin 256 → ℝ := fun i0 i1 => (parameter2 i0 i1 * (1/16))
    let vmul_3 : Fin 320 → Fin 512 → ℝ := fun i0 i1 => (parameter3 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_4 : Fin 256 → ℝ := fun i0 => (parameter4 i0 * (1/10))
    let vadd_1 : Fin 256 → ℝ := fun i0 => (vmul_4 i0 + 1)
    let vmul_5 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => (parameter5 i0 i1 * (1/16))
    let vmul_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => (parameter6 i0 i1 * (1/32))
    let vto_copy : Fin 256 → Fin 512 → ℝ := fun i0 i1 => vmul i0 i1
    let vto_copy_1 : Fin 256 → ℝ := fun i0 => vadd i0
    let vto_copy_2 : Fin 1536 → Fin 256 → ℝ := fun i0 i1 => vmul_2 i0 i1
    let vto_copy_3 : Fin 320 → Fin 512 → ℝ := fun i0 i1 => vmul_3 i0 i1
    let vto_copy_4 : Fin 256 → ℝ := fun i0 => vadd_1 i0
    let vto_copy_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => vmul_6 i0 i1
    let vcat : Fin 128 → Int := fun i0 => (if i0.val < 48 then (0 + ((i0.val - 0) : Int) * 1) else (if i0.val < 64 then (0 + ((i0.val - 48) : Int) * 1) else (0 + ((i0.val - 64) : Int) * 1)))
    let vdiv : Fin 32 → ℝ := fun i0 => ((0 + (i0.val : ℝ) * 2) / 64)
    let vpow_1 : Fin 32 → ℝ := fun i0 => Real.rpow ((10000 : ℝ)) (vdiv i0)
    let vreciprocal : Fin 32 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul_7 : Fin 32 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vmul_8 : Fin 128 → Fin 32 → ℝ := fun i0 i1 => ((0 + (i0.val : ℝ) * 1) * vmul_7 i1)
    let vcat_1 : Fin 128 → Fin 64 → ℝ := fun i0 i1 => (if i1.val < 32 then vmul_8 i0 ⟨((i1.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vmul_8 i0 ⟨((i1.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcos : Fin 128 → Fin 64 → ℝ := fun i0 i1 => (Real.cos (vcat_1 i0 i1))
    let vindex : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vcos ⟨(Int.toNat ((((((((i0.val * 128) : Int) + vcat i1) * (64 : Int)) + (i2.val : Int)) / (64 : Int)) % (128 : Int)))) % 128, Nat.mod_lt _ (by decide)⟩ ⟨(Int.toNat (((((((i0.val * 128) : Int) + vcat i1) * (64 : Int)) + (i2.val : Int)) % (64 : Int)))) % 64, Nat.mod_lt _ (by decide)⟩
    let vsin : Fin 128 → Fin 64 → ℝ := fun i0 i1 => (Real.sin (vcat_1 i0 i1))
    let vindex_1 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vsin ⟨(Int.toNat ((((((((i0.val * 128) : Int) + vcat i1) * (64 : Int)) + (i2.val : Int)) / (64 : Int)) % (128 : Int)))) % 128, Nat.mod_lt _ (by decide)⟩ ⟨(Int.toNat (((((((i0.val * 128) : Int) + vcat i1) * (64 : Int)) + (i2.val : Int)) % (64 : Int)))) % 64, Nat.mod_lt _ (by decide)⟩
    let vto_copy_14 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vindex ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_15 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vindex_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vsub : Fin 1 → Fin 1 → Int := fun i0 i1 => (vcat ⟨(0) % 128, Nat.mod_lt _ (by decide)⟩ - 1)
    let vcat_2 : Fin 1 → Fin 129 → Int := fun i0 i1 => (if i1.val < 1 then vsub i0 ⟨((i1.val - 0)) % 1, Nat.mod_lt _ (by decide)⟩ else vcat ⟨((((i0.val * 128) + (i1.val - 1)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vsub_1 : Fin 1 → Fin 128 → Int := fun i0 i1 => (vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((i1.val + 1)) % 129, Nat.mod_lt _ (by decide)⟩ - vcat_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(i1.val) % 129, Nat.mod_lt _ (by decide)⟩)
    let vne : Fin 1 → Fin 128 → Bool := fun i0 i1 => decide (vsub_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ≠ 1)
    let vcumsum_1 : Fin 1 → Fin 128 → Int := fun i0 i1 => (∑ lane : Fin 128, if lane.val ≤ i1.val then (if vne i0 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ then (1 : Int) else 0) else 0)
    let veq : Fin 1 → Fin 1 → Fin 128 → Fin 128 → Bool := fun i0 i1 i2 i3 => decide (vcumsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 = vcumsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i3)
    let vtril : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≤ i0.val + 0 then Bool.true else Bool.false)
    let vbitwise_and : Fin 1 → Fin 1 → Fin 128 → Fin 128 → Bool := fun i0 i1 i2 i3 => (veq ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 && vtril i2 i3)
    let vwhere : Fin 1 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vbitwise_and ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 then 0 else (-338953138925153550000000000000000000000 : ℝ))
    let vto_copy_18 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => vmul_5 i0 i1
    let vmm : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy i1 lane)
    let vto_copy_19 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm i1 i2
    let vpow_2 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_2 i0 i1 lane) / 256)
    let vadd_2 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_9 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_10 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_20 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_1 : Fin 128 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 128 → Fin 320 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_3 i1 lane)
    let vto_copy_21 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_2 i1 ⟨(i2.val) % 320, Nat.mod_lt _ (by decide)⟩
    let vpow_3 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_3 i0 i1 lane) / 256)
    let vadd_3 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_11 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_12 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_11 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_22 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_12 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_3 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_18 i1 lane)
    let vmul_13 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_14 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub_4 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_13 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 - vmul_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vmul_15 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_16 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 i2 ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_4 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_16 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vcat_4 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_4 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_4 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vmul_17 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_18 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub_5 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_19 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_20 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 i2 ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_5 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_5 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_5 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_5 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcat_6 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_1 ⟨(((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 1536) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 1536)) % 1536, Nat.mod_lt _ (by decide)⟩ else vcat_4 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_7 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_3 ⟨(((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 2048) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ else vcat_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vbmm : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vcat_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i1 lane * vcat_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i2 lane)
    let vmul_21 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm i1 i2 i3 * (3608439182435161/50000000000000000))
    let vadd_6 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vwhere ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy_23 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vsoftmax : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_23 i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy_23 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_24 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmm_5 : Fin 128 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_6 lane i1)
    let vbmm_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_24 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmm_5 lane ⟨(((i0.val * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_3 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_5 i1 ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vmm_3 i2 ⟨(((i0.val * 256) + lane.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_25 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vbmm_3 i1 i2 i3
    let vsoftmax_backward_data : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_25 i0 i1 i2 i3 - (∑ lane : Fin 128, vto_copy_25 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2 i3
    let vto_copy_26 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax_backward_data ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmul_22 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 * (3608439182435161/50000000000000000))
    let vbmm_4 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vcat_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmul_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i2)
    let vcat_8 : Fin 1 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_4 ⟨(((((i0.val * 196608) + (i1.val * 24576) + ((i3.val - 0) * 128) + i2.val) / 24576) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ i2 else vbmm_2 ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 16384) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 128) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vmm_6 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 128, vcat_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((lane.val * 2048) + i0.val) / 256) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane ⟨((((lane.val * 2048) + i0.val) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ * vto_copy_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ lane i1)
    vmm_6 i0 i1

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_varlen_wgrad
