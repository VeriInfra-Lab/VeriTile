import bench.pretrain_kernels.tasks.local_mla_fa4_absorb_bi_linear_dx.IO
import Mathlib.Analysis.SpecialFunctions.Pow.Real

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_bi_linear_dx
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
    let input1 := xs.input1
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
    let vtriu : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≥ i0.val + 1 then Bool.true else Bool.false)
    let vmasked_fill : Fin 2 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vtriu i2 i3 then (-338953138925153550000000000000000000000 : ℝ) else 0)
    let vmm : Fin 256 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy i1 lane)
    let vto_copy_16 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vpow_1 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_16 i0 i1 i2) * (vto_copy_16 i0 i1 i2))
    let vmean : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_1 i0 i1 lane) / 256)
    let vadd_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_7 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_16 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_8 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_7 i0 i1 i2)
    let vto_copy_17 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_8 i0 i1 i2
    let vmm_1 : Fin 256 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_17 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 256 → Fin 576 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_3 i1 lane)
    let vto_copy_18 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmm_2 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(i2.val) % 576, Nat.mod_lt _ (by decide)⟩
    let vpow_2 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => ((vto_copy_18 i0 i1 i2) * (vto_copy_18 i0 i1 i2))
    let vmean_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 512, vpow_2 i0 i1 lane) / 512)
    let vadd_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_9 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_18 i0 i1 i2 * vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_10 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_9 i0 i1 i2)
    let vto_copy_19 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmul_10 i0 i1 i2
    let vmm_3 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 512, vto_copy_19 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_5 i1 lane)
    let vmul_11 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_12 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * 0)
    let vsub : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_11 i0 i1 i2 i3 - vmul_12 i0 i1 i2 i3)
    let vmul_13 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 129)) % 1536, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_14 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 192) + (i3.val * 2) + 128)) % 1536, Nat.mod_lt _ (by decide)⟩ * 0)
    let vadd_4 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_13 i0 i1 i2 i3 + vmul_14 i0 i1 i2 i3)
    let vcat : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_4 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vmul_15 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 512)) % 576, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_16 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 513)) % 576, Nat.mod_lt _ (by decide)⟩ * 0)
    let vsub_1 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_15 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_16 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_17 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 513)) % 576, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_18 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 512)) % 576, Nat.mod_lt _ (by decide)⟩ * 0)
    let vadd_5 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_17 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_18 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_1 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_1 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_5 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcat_2 : Fin 2 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_1 ⟨(((((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 196608) % 2) * 128) + ((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) / 1536) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i2.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 1536)) % 1536, Nat.mod_lt _ (by decide)⟩ else vcat i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_3 : Fin 2 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vmm_3 ⟨(((((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 262144) % 2) * 128) + ((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 2048) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ else vcat_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vbmm : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vcat_2 ⟨(((((i0.val * 24576) + (i1.val * 192) + lane.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * vcat_3 ⟨(((((i0.val * 24576) + (lane.val * 128) + i2.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i2 lane)
    let vmul_19 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 i3 * (3608439182435161/50000000000000000))
    let vadd_6 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul_19 i0 i1 i2 i3 + vmasked_fill i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vto_copy_20 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd_6 i0 i1 i2 i3
    let vsoftmax : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_20 i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy_20 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_21 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax i0 i1 i2 i3
    let vmm_5 : Fin 256 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input1 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_6 lane i1)
    let vbmm_2 : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_21 ⟨(((((i0.val * 16384) + (lane.val * 128) + i1.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane i1 * vmm_5 ⟨(((((((i0.val * 16384) + (lane.val * 128) + i2.val) / 131072) % 2) * 128) + lane.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val % 8) * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_3 : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_5 ⟨(((((((i0.val * 16384) + (i1.val * 128) + lane.val) / 131072) % 2) * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val % 8) * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vmm_3 ⟨(((((((i0.val * 16384) + (i2.val * 128) + lane.val) / 131072) % 2) * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val % 8) * 256) + lane.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_22 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vbmm_3 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 i3
    let vsoftmax_backward_data : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_22 i0 i1 i2 i3 - (∑ lane : Fin 128, vto_copy_22 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2 i3
    let vto_copy_23 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax_backward_data i0 i1 i2 i3
    let vmul_20 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_23 i0 i1 i2 i3 * (3608439182435161/50000000000000000))
    let vbmm_4 : Fin 16 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vcat_2 ⟨(((((i0.val * 24576) + (lane.val * 192) + i1.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane i1 * vmul_20 ⟨(((((i0.val * 16384) + (lane.val * 128) + i2.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane i2)
    let vbmm_5 : Fin 16 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmul_20 ⟨(((((i0.val * 16384) + (i1.val * 128) + lane.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * vcat_3 ⟨(((((i0.val * 24576) + (i2.val * 128) + lane.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ lane i2)
    let vsum_1 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 8, vbmm_4 ⟨(((i0.val * 8) + lane.val)) % 16, Nat.mod_lt _ (by decide)⟩ ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ i2)
    let vmul_21 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * 0)
    let vmul_22 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * 1)
    let vneg : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩))
    let vmul_23 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 * 0)
    let vadd_7 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_22 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_23 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_24 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩ * 1)
    let vadd_8 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_21 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_24 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_25 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * 0)
    let vmul_26 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * 1)
    let vneg_1 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vbmm_5 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩))
    let vmul_27 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg_1 i0 i1 i2 i3 * 0)
    let vadd_9 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_26 i0 i1 i2 i3 + vmul_27 i0 i1 i2 i3)
    let vmul_28 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ * 1)
    let vadd_10 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_25 i0 i1 i2 i3 + vmul_28 i0 i1 i2 i3)
    let vslice_backward : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_7 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_8 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_11 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vslice_backward_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vslice_backward_2 : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_9 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_3 : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_10 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_12 : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward_2 i0 i1 i2 i3 + vslice_backward_3 i0 i1 i2 i3)
    let vcat_4 : Fin 2 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_4 ⟨(((((i0.val * 196608) + (i1.val * 24576) + ((i3.val - 0) * 128) + i2.val) / 24576) % 16)) % 16, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ i2 else vbmm_2 ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 16384) % 16)) % 16, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 128) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_5 : Fin 2 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_5 ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 24576) % 16)) % 16, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 192) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ else vadd_12 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_6 : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vcat_4 ⟨(((((i0.val * 2048) + lane.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨(((((((((i0.val * 2048) + lane.val) / 262144) % 2) * 262144) + ((i0.val % 128) * 2048) + lane.val) / 256) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((((((i0.val * 2048) + lane.val) / 262144) % 2) * 262144) + ((i0.val % 128) * 2048) + lane.val) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 lane i1)
    let vto_copy_24 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmm_6 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vmul_29 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_24 i0 i1 i2 * vto_copy_4 i2)
    let vmul_30 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_29 i0 i1 i2 * vto_copy_18 i0 i1 i2)
    let vmul_31 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_29 i0 i1 i2 * vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 512, vmul_30 i0 i1 lane)
    let vpow_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_32 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_33 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_32 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_33 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 512)
    let vpow_4 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => ((vto_copy_18 i0 i1 i2))
    let vmul_34 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vpow_4 i0 i1 i2 * (2 : ℝ))
    let vmul_35 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vdiv i0 i1 i2 * vmul_34 i0 i1 i2)
    let vadd_13 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmul_31 i0 i1 i2 + vmul_35 i0 i1 i2)
    let vto_copy_25 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vadd_13 i0 i1 i2
    let vcat_6 : Fin 2 → Fin 128 → Fin 576 → ℝ := fun i0 i1 i2 => (if i2.val < 512 then vto_copy_25 i0 i1 ⟨((i2.val - 0)) % 512, Nat.mod_lt _ (by decide)⟩ else vadd_11 ⟨(((((i0.val * 8192) + (i1.val * 64) + (i2.val - 512)) / 8192) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 8192) + (i1.val * 64) + (i2.val - 512)) / 64) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 8192) + (i1.val * 64) + (i2.val - 512)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_7 : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 576, vcat_6 ⟨(((((i0.val * 576) + lane.val) / 73728) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_3 lane i1)
    let vmm_8 : Fin 256 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 1536, vcat_5 ⟨(((((i0.val * 1536) + lane.val) / 196608) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨(((((((((i0.val * 1536) + lane.val) / 196608) % 2) * 196608) + ((i0.val % 128) * 1536) + lane.val) / 192) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((((((i0.val * 1536) + lane.val) / 196608) % 2) * 196608) + ((i0.val % 128) * 1536) + lane.val) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_2 lane i1)
    let vto_copy_26 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_8 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vmul_36 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_26 i0 i1 i2 * vto_copy_1 i2)
    let vmul_37 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_36 i0 i1 i2 * vto_copy_16 i0 i1 i2)
    let vmul_38 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_36 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vmul_37 i0 i1 lane)
    let vpow_5 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_39 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_40 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_39 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_5 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_1 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_40 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 256)
    let vpow_6 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_16 i0 i1 i2))
    let vmul_41 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vpow_6 i0 i1 i2 * (2 : ℝ))
    let vmul_42 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vdiv_1 i0 i1 i2 * vmul_41 i0 i1 i2)
    let vadd_14 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_38 i0 i1 i2 + vmul_42 i0 i1 i2)
    let vto_copy_27 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vadd_14 i0 i1 i2
    let vmm_9 : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_27 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy lane i1)
    let vadd_15 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmm_7 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2 + vmm_9 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2)
    vadd_15 i0 i1 i2

end Contract
end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_bi_linear_dx
