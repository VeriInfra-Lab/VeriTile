import bench.pretrain_kernels.tasks.local_mla_fa4_nonabsorb_varlen_dx.Contract

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_varlen_dx
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2) =>
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
    let vto_copy_5 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => vmul_5 i0 i1
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
    let vmm : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy i1 lane)
    let vto_copy_18 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm i1 i2
    let vpow_2 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_2 i0 i1 lane) / 256)
    let vadd_2 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_9 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_10 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_19 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_1 : Fin 128 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_19 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 128 → Fin 320 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_3 i1 lane)
    let vto_copy_20 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_2 i1 ⟨(i2.val) % 320, Nat.mod_lt _ (by decide)⟩
    let vpow_3 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2) * (vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmean_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_3 i0 i1 lane) / 256)
    let vadd_3 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_11 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_12 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_11 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_21 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_12 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_3 : Fin 128 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_21 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_5 i1 lane)
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
    let vto_copy_22 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vadd_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vsoftmax : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_22 i0 i1 i2 i3) / (∑ lane : Fin 128, Real.exp (vto_copy_22 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_23 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmm_5 : Fin 128 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_6 lane i1)
    let vbmm_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_23 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmm_5 lane ⟨(((i0.val * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_3 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_5 i1 ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vmm_3 i2 ⟨(((i0.val * 256) + lane.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_24 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vbmm_3 i1 i2 i3
    let vsoftmax_backward_data : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_24 i0 i1 i2 i3 - (∑ lane : Fin 128, vto_copy_24 i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 i2 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2 i3
    let vto_copy_25 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => vsoftmax_backward_data ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3
    let vmul_22 : Fin 1 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vto_copy_25 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 * (3608439182435161/50000000000000000))
    let vbmm_4 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vcat_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i1 * vmul_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i2)
    let vbmm_5 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmul_22 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 i1 lane * vcat_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane i2)
    let vsum_1 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (∑ lane : Fin 8, vbmm_4 ⟨(((i0.val * 8) + lane.val)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ i2)
    let vmul_23 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_24 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val + 32)) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vneg : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩))
    let vmul_25 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_7 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_24 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_25 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_26 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vsum_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_8 : Fin 1 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_23 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_27 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 i1 i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmul_28 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 i1 i2 ⟨((i3.val + 160)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vneg_1 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (- (vbmm_5 i1 i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩))
    let vmul_29 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vneg_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 * vto_copy_15 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_9 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_28 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_29 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vmul_30 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vbmm_5 i1 i2 ⟨((i3.val + 128)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(i3.val) % 64, Nat.mod_lt _ (by decide)⟩)
    let vadd_10 : Fin 1 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_27 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vmul_30 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vslice_backward : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_7 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_1 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_8 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_11 : Fin 1 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vslice_backward_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vslice_backward_2 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 1 ∧ i3.val < 64 ∧ (i3.val - 1) % 2 = 0) then vadd_9 i0 i1 i2 ⟨(((i3.val - 1) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_3 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if (i3.val ≥ 0 ∧ i3.val < 64 ∧ (i3.val - 0) % 2 = 0) then vadd_10 i0 i1 i2 ⟨(((i3.val - 0) / 2)) % 32, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_12 : Fin 1 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (vslice_backward_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3 + vslice_backward_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 i3)
    let vcat_8 : Fin 1 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_4 ⟨(((((i0.val * 196608) + (i1.val * 24576) + ((i3.val - 0) * 128) + i2.val) / 24576) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 1536) + (i1.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ i2 else vbmm_2 ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 16384) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) / 128) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 131072) + (i1.val * 16384) + (i2.val * 128) + (i3.val - 128)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vcat_9 : Fin 1 → Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vbmm_5 ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 24576) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) / 192) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 196608) + (i1.val * 24576) + (i2.val * 192) + (i3.val - 0)) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ else vadd_12 i0 i1 i2 ⟨((i3.val - 128)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_6 : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vcat_8 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 2048) + lane.val) / 256) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i0 ⟨((((i0.val * 2048) + lane.val) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ * vto_copy_5 lane i1)
    let vto_copy_26 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_6 i1 i2
    let vmul_31 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_26 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_4 i2)
    let vmul_32 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_31 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vmul_33 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_31 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_2 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vmul_32 i0 i1 lane)
    let vpow_4 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_34 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_35 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_34 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_4 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_1 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_35 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 256)
    let vpow_5 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_36 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vpow_5 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (2 : ℝ))
    let vmul_37 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vdiv_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vmul_36 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vadd_13 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_33 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 + vmul_37 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_27 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vadd_13 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vcat_10 : Fin 1 → Fin 128 → Fin 320 → ℝ := fun i0 i1 i2 => (if i2.val < 256 then vto_copy_27 i0 i1 ⟨((i2.val - 0)) % 256, Nat.mod_lt _ (by decide)⟩ else vadd_11 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 8192) + (i1.val * 64) + (i2.val - 256)) / 64) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 8192) + (i1.val * 64) + (i2.val - 256)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vmm_7 : Fin 128 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 320, vcat_10 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy_3 lane i1)
    let vmm_8 : Fin 128 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 1536, vcat_9 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 1536) + lane.val) / 192) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i0 ⟨((((i0.val * 1536) + lane.val) % 192)) % 192, Nat.mod_lt _ (by decide)⟩ * vto_copy_2 lane i1)
    let vto_copy_28 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_8 i1 i2
    let vmul_38 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_28 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_1 i2)
    let vmul_39 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_38 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vmul_40 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_38 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vsum_3 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vmul_39 i0 i1 lane)
    let vpow_6 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) * (vrsqrt ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_41 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vsum_3 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * (-1/2))
    let vmul_42 : Fin 1 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmul_41 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ * vpow_6 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vdiv_2 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_42 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ / 256)
    let vpow_7 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_18 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2))
    let vmul_43 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vpow_7 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * (2 : ℝ))
    let vmul_44 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vdiv_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 * vmul_43 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vadd_14 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vmul_40 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2 + vmul_44 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2)
    let vto_copy_29 : Fin 1 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vadd_14 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i1 i2
    let vmm_9 : Fin 128 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_29 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i0 lane * vto_copy lane i1)
    let vadd_15 : Fin 1 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vmm_7 i1 i2 + vmm_9 i1 i2)
    vadd_15 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_nonabsorb_varlen_dx
