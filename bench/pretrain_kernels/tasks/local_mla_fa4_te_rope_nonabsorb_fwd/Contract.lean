import bench.pretrain_kernels.tasks.local_mla_fa4_te_rope_nonabsorb_fwd.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_nonabsorb_fwd
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
    let parameter1 := xs.parameter1
    let parameter2 := xs.parameter2
    let parameter3 := xs.parameter3
    let parameter4 := xs.parameter4
    let parameter5 := xs.parameter5
    let parameter6 := xs.parameter6
    let input0 := xs.input0
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
    let vto_copy_5 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => vmul_5 i0 i1
    let vto_copy_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => vmul_6 i0 i1
    let vto_copy_14 : Fin 32 → ℝ := fun i0 => ((0 + (i0.val : Int) * 2) : ℝ)
    let vdiv : Fin 32 → ℝ := fun i0 => (vto_copy_14 i0 / 64)
    let vpow_1 : Fin 32 → ℝ := fun i0 => Real.rpow ((10000 : ℝ)) (vdiv i0)
    let vreciprocal : Fin 32 → ℝ := fun i0 => (1 / (vpow_1 i0))
    let vmul_7 : Fin 32 → ℝ := fun i0 => (vreciprocal i0 * (1 : ℝ))
    let vto_copy_16 : Fin 1 → Fin 1 → Fin 128 → ℝ := fun i0 i1 i2 => ((0 + (i2.val : Int) * 1) : ℝ)
    let vbmm : Fin 1 → Fin 32 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 1, vmul_7 ⟨(((i0.val * 32) + i1.val + lane.val)) % 32, Nat.mod_lt _ (by decide)⟩ * vto_copy_16 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + (lane.val * 128) + i2.val)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (if i2.val < 32 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 32) + (i2.val - 0)) % 32)) % 32, Nat.mod_lt _ (by decide)⟩ i1 else vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 32) + (i2.val - 32)) % 32)) % 32, Nat.mod_lt _ (by decide)⟩ i1)
    let vcos : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (Real.cos (vcat ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_9 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (vcos ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vsin : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (Real.sin (vcat ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_10 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => (vsin ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (1 : ℝ))
    let vto_copy_17 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vmul_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vto_copy_18 : Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 => vmul_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vtriu : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≥ i0.val + 1 then Bool.true else Bool.false)
    let vmasked_fill : Fin 2 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vtriu i2 i3 then (-338953138925153547590470800371487866880 : ℝ) else 0)
    let vmm : Fin 256 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy i1 lane)
    let vto_copy_20 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vpow_3 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 i0 i1 i2) * (vto_copy_20 i0 i1 i2))
    let vmean : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_3 i0 i1 lane) / 256)
    let vadd_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_13 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_20 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_14 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_13 i0 i1 i2)
    let vto_copy_21 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_14 i0 i1 i2
    let vmm_1 : Fin 256 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_21 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 256 → Fin 320 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_3 i1 lane)
    let vto_copy_22 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_2 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(i2.val) % 320, Nat.mod_lt _ (by decide)⟩
    let vpow_4 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_22 i0 i1 i2) * (vto_copy_22 i0 i1 i2))
    let vmean_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_4 i0 i1 lane) / 256)
    let vadd_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_15 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_22 i0 i1 i2 * vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_16 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_15 i0 i1 i2)
    let vto_copy_23 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_16 i0 i1 i2
    let vmm_3 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_23 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_5 i1 lane)
    let vmul_17 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_18 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_17 i0 i1 i2 i3 - vmul_18 i0 i1 i2 i3)
    let vmul_19 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_20 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_4 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_19 i0 i1 i2 i3 + vmul_20 i0 i1 i2 i3)
    let vcat_2 : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_4 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vmul_21 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_22 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vsub_1 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_21 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_22 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_23 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_17 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_24 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_5 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_23 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_24 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_3 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_1 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_5 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcat_4 : Fin 2 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_1 ⟨(((((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 196608) % 2) * 128) + ((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 1536) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 1536)) % 1536, Nat.mod_lt _ (by decide)⟩ else vcat_2 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_5 : Fin 2 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_3 ⟨(((((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 262144) % 2) * 128) + ((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 2048) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ else vcat_3 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vbmm_2 : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vcat_4 ⟨(((((i0.val * 24576) + (i1.val * 192) + lane.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * vcat_5 ⟨(((((i0.val * 24576) + (lane.val * 128) + i2.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i2 lane)
    let vmul_25 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm_2 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 i3 * (3608439182435161/50000000000000000))
    let vadd_6 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul_25 i0 i1 i2 i3 + vmasked_fill i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy_24 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd_6 i0 i1 i2 i3
    let vsoftmax : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_24 i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy_24 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_25 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax i0 i1 i2 i3
    let vbmm_3 : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_25 ⟨(((((i0.val * 16384) + (i1.val * 128) + lane.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * vmm_3 ⟨(((((((i0.val * 16384) + (lane.val * 128) + i2.val) / 131072) % 2) * 128) + lane.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val % 8) * 256) + i2.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩)
    let vmm_4 : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 1024, vbmm_3 ⟨(((((((i0.val * 1024) + lane.val) / 131072) % 2) * 8) + ((((((((i0.val * 1024) + lane.val) / 131072) % 2) * 131072) + ((i0.val % 128) * 1024) + lane.val) / 128) % 8))) % 16, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((((((i0.val * 1024) + lane.val) / 131072) % 2) * 131072) + ((i0.val % 128) * 1024) + lane.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 i1 lane)
    vmm_4 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_te_rope_nonabsorb_fwd
