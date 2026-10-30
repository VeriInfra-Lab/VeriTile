import bench.pretrain_kernels.tasks.local_mla_fa4_absorb_vs_nonabsorb_wgrad.Contract

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_vs_nonabsorb_wgrad
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1) =>
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
    let vto_copy_6 : Fin 512 → Fin 1024 → ℝ := fun i0 i1 => vmul_6 i0 i1
    let vto_copy_16 : Fin 2048 → Fin 512 → ℝ := fun i0 i1 => vmul_5 i0 i1
    let vmm : Fin 256 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy i1 lane)
    let vto_copy_17 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vpow_1 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_17 i0 i1 i2) * (vto_copy_17 i0 i1 i2))
    let vmean : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_1 i0 i1 lane) / 256)
    let vadd_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_7 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_17 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_8 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_7 i0 i1 i2)
    let vto_copy_18 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_8 i0 i1 i2
    let vmm_1 : Fin 256 → Fin 1536 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_18 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 256 → Fin 576 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_3 i1 lane)
    let vto_copy_19 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmm_2 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(i2.val) % 576, Nat.mod_lt _ (by decide)⟩
    let vpow_2 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => ((vto_copy_19 i0 i1 i2) * (vto_copy_19 i0 i1 i2))
    let vmean_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 512, vpow_2 i0 i1 lane) / 512)
    let vadd_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_9 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_19 i0 i1 i2 * vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_10 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_9 i0 i1 i2)
    let vto_copy_20 : Fin 2 → Fin 128 → Fin 512 → ℝ := fun i0 i1 i2 => vmul_10 i0 i1 i2
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
    let vbmm : Fin 1 → Fin 256 → Fin 1024 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 512, vto_copy_20 ⟨(((((i0.val * 131072) + (i1.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 256) + i1.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_16 ⟨(((((((i0.val * 524288) + (lane.val * 1024) + i2.val) / 128) % 8) * 256) + (((i0.val * 524288) + (lane.val * 1024) + i2.val) % 128) + i0.val)) % 2048, Nat.mod_lt _ (by decide)⟩ lane)
    let vbmm_1 : Fin 1 → Fin 256 → Fin 1024 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 512, vto_copy_20 ⟨(((((i0.val * 131072) + (i1.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 256) + i1.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_16 ⟨(((((((i0.val * 524288) + (lane.val * 1024) + i2.val) / 128) % 8) * 256) + (((i0.val * 524288) + (lane.val * 1024) + i2.val) % 128) + i0.val + 128)) % 2048, Nat.mod_lt _ (by decide)⟩ lane)
    let vcat_2 : Fin 256 → Fin 8 → Fin 192 → ℝ := fun i0 i1 i2 => (if i2.val < 128 then vmm_1 ⟨(((((((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) / 131072) % 2) * 128) + ((((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) / 1024) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((((((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) / 128) % 8) * 192) + (((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) % 128))) % 1536, Nat.mod_lt _ (by decide)⟩ else vcat ⟨(((((i0.val * 512) + (i1.val * 64) + (i2.val - 128)) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 512) + (i1.val * 64) + (i2.val - 128)) / 64) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 512) + (i1.val * 64) + (i2.val - 128)) / 512) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 512) + (i1.val * 64) + (i2.val - 128)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_3 : Fin 256 → Fin 8 → Fin 192 → ℝ := fun i0 i1 i2 => (if i2.val < 128 then vbmm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) / 1024) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((((((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) / 128) % 8) * 128) + (((i0.val * 1024) + (i1.val * 128) + (i2.val - 0)) % 128))) % 1024, Nat.mod_lt _ (by decide)⟩ else vcat_1 ⟨(((((i0.val * 64) + (i2.val - 128)) / 8192) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 64) + (i2.val - 128)) / 64) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 64) + (i2.val - 128)) % 64)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_21 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vcat_2 ⟨(i1.val) % 256, Nat.mod_lt _ (by decide)⟩ i0 i2
    let vto_copy_22 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vcat_3 ⟨(i1.val) % 256, Nat.mod_lt _ (by decide)⟩ i0 i2
    let vbmm_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vto_copy_21 i0 i1 lane * vto_copy_22 i0 i2 lane)
    let vmul_19 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vbmm_2 i0 i1 i2 * (3608439182435161/50000000000000000))
    let vadd_6 : Fin 128 → Fin 1 → Int := fun i0 i1 => ((0 + (i0.val : Int) * 1) + 128)
    let vsub_2 : Fin 128 → Fin 1 → Int := fun i0 i1 => (vadd_6 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ - 128)
    let vgt : Fin 128 → Fin 128 → Bool := fun i0 i1 => decide ((0 + (i1.val : Int) * 1) > vsub_2 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmasked_fill_2 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vgt i1 i2 then 0 else vmul_19 i0 i1 i2)
    let vmasked_fill_2Finite : Fin 8 → Fin 128 → Fin 128 → Prop := fun i0 i1 i2 => (if vgt i1 i2 then False else True)
    let vsoftmax : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vmasked_fill_2Finite i0 i1 i2 then Real.exp (vmasked_fill_2 i0 i1 i2) else 0) / (∑ lane : Fin 128, (if vmasked_fill_2Finite i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ then Real.exp (vmasked_fill_2 i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩) else 0))
    let vto_copy_23 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vsoftmax i0 i1 i2
    let vto_copy_24 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vcat_2 ⟨((i1.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ i0 i2
    let vto_copy_25 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vcat_3 ⟨((i1.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ i0 i2
    let vbmm_4 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 192, vto_copy_24 i0 i1 lane * vto_copy_25 i0 i2 lane)
    let vmul_20 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vbmm_4 i0 i1 i2 * (3608439182435161/50000000000000000))
    let vadd_7 : Fin 128 → Fin 1 → Int := fun i0 i1 => ((0 + (i0.val : Int) * 1) + 128)
    let vsub_3 : Fin 128 → Fin 1 → Int := fun i0 i1 => (vadd_7 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ - 128)
    let vgt_1 : Fin 128 → Fin 128 → Bool := fun i0 i1 => decide ((0 + (i1.val : Int) * 1) > vsub_3 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmasked_fill_3 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vgt_1 i1 i2 then 0 else vmul_20 i0 i1 i2)
    let vmasked_fill_3Finite : Fin 8 → Fin 128 → Fin 128 → Prop := fun i0 i1 i2 => (if vgt_1 i1 i2 then False else True)
    let vsoftmax_1 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vmasked_fill_3Finite i0 i1 i2 then Real.exp (vmasked_fill_3 i0 i1 i2) else 0) / (∑ lane : Fin 128, (if vmasked_fill_3Finite i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ then Real.exp (vmasked_fill_3 i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩) else 0))
    let vto_copy_26 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vsoftmax_1 i0 i1 i2
    let vmm_4 : Fin 256 → Fin 1024 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input1 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_6 lane i1)
    let vbmm_6 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_26 i0 lane i1 * vmm_4 ⟨((lane.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_7 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_4 ⟨((i1.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vbmm_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨((i2.val + 128)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_27 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vbmm_7 i0 i1 i2
    let vsoftmax_backward_data : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vto_copy_27 i0 i1 i2 - (∑ lane : Fin 128, vto_copy_27 i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax_1 i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax_1 i0 i1 i2
    let vmasked_fill_4 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vgt_1 i1 i2 then 0 else vsoftmax_backward_data i0 i1 i2)
    let vmul_21 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vmasked_fill_4 i0 i1 i2 * (3608439182435161/50000000000000000))
    let vbmm_8 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_24 i0 lane i1 * vmul_21 i0 lane i2)
    let vslice_backward : Fin 256 → Fin 8 → Fin 128 → ℝ := fun i0 i1 i2 => (if (i0.val ≥ 128 ∧ i0.val < 256 ∧ (i0.val - 128) % 1 = 0) then vbmm_6 ⟨(((((i1.val * 16384) + ((i0.val - 128) * 128) + i2.val) / 16384) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i1.val * 128) + (i0.val - 128)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i2 else 0)
    let vto_copy_28 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vbmm_8 i0 i2 i1
    let vslice_backward_1 : Fin 256 → Fin 8 → Fin 192 → ℝ := fun i0 i1 i2 => (if (i0.val ≥ 128 ∧ i0.val < 256 ∧ (i0.val - 128) % 1 = 0) then vto_copy_28 i1 ⟨((i0.val - 128)) % 128, Nat.mod_lt _ (by decide)⟩ i2 else 0)
    let vbmm_9 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_23 i0 lane i1 * vmm_4 ⟨(lane.val) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + i2.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vbmm_10 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vmm_4 ⟨(i1.val) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩ * vbmm_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(i2.val) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i0.val * 128) + lane.val)) % 1024, Nat.mod_lt _ (by decide)⟩)
    let vto_copy_29 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => vbmm_10 i0 i1 i2
    let vsoftmax_backward_data_1 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vto_copy_29 i0 i1 i2 - (∑ lane : Fin 128, vto_copy_29 i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩ * vsoftmax i0 i1 ⟨(lane.val) % 128, Nat.mod_lt _ (by decide)⟩)) * vsoftmax i0 i1 i2
    let vmasked_fill_5 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (if vgt i1 i2 then 0 else vsoftmax_backward_data_1 i0 i1 i2)
    let vmul_22 : Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (vmasked_fill_5 i0 i1 i2 * (3608439182435161/50000000000000000))
    let vbmm_11 : Fin 8 → Fin 192 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_21 i0 lane i1 * vmul_22 i0 lane i2)
    let vslice_backward_2 : Fin 256 → Fin 8 → Fin 128 → ℝ := fun i0 i1 i2 => (if (i0.val ≥ 0 ∧ i0.val < 128 ∧ (i0.val - 0) % 1 = 0) then vbmm_9 ⟨(((((i1.val * 16384) + ((i0.val - 0) * 128) + i2.val) / 16384) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i1.val * 128) + (i0.val - 0)) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i2 else 0)
    let vadd_8 : Fin 256 → Fin 8 → Fin 128 → ℝ := fun i0 i1 i2 => (vslice_backward i0 i1 i2 + vslice_backward_2 i0 i1 i2)
    let vto_copy_30 : Fin 8 → Fin 128 → Fin 192 → ℝ := fun i0 i1 i2 => vbmm_11 i0 i2 i1
    let vslice_backward_3 : Fin 256 → Fin 8 → Fin 192 → ℝ := fun i0 i1 i2 => (if (i0.val ≥ 0 ∧ i0.val < 128 ∧ (i0.val - 0) % 1 = 0) then vto_copy_30 i1 ⟨((i0.val - 0)) % 128, Nat.mod_lt _ (by decide)⟩ i2 else 0)
    let vadd_9 : Fin 256 → Fin 8 → Fin 192 → ℝ := fun i0 i1 i2 => (vslice_backward_1 i0 i1 i2 + vslice_backward_3 i0 i1 i2)
    let vbmm_12 : Fin 1 → Fin 512 → Fin 1024 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vto_copy_20 ⟨(((((i0.val * 131072) + (lane.val * 512) + i1.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 256) + lane.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1 * vadd_8 ⟨(((i0.val * 256) + lane.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 262144) + (lane.val * 1024) + i2.val) / 128) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (lane.val * 1024) + i2.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩)
    let vbmm_13 : Fin 1 → Fin 512 → Fin 1024 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vto_copy_20 ⟨(((((i0.val * 131072) + (lane.val * 512) + i1.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 256) + lane.val) % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1 * vadd_9 ⟨(((i0.val * 256) + lane.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((((i0.val * 262144) + (lane.val * 1024) + i2.val) / 128) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (lane.val * 1024) + i2.val) % 128)) % 192, Nat.mod_lt _ (by decide)⟩)
    let vslice_backward_4 : Fin 8 → Fin 256 → Fin 512 → ℝ := fun i0 i1 i2 => (if (i1.val ≥ 128 ∧ i1.val < 256 ∧ (i1.val - 128) % 1 = 0) then vbmm_12 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(((((((i0.val * 65536) + ((i1.val - 128) * 512) + i2.val) / 65536) % 8) * 128) + (((i0.val * 128) + (i1.val - 128)) % 128))) % 1024, Nat.mod_lt _ (by decide)⟩ else 0)
    let vslice_backward_5 : Fin 8 → Fin 256 → Fin 512 → ℝ := fun i0 i1 i2 => (if (i1.val ≥ 0 ∧ i1.val < 128 ∧ (i1.val - 0) % 1 = 0) then vbmm_13 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨(((((((i0.val * 65536) + ((i1.val - 0) * 512) + i2.val) / 65536) % 8) * 128) + (((i0.val * 128) + (i1.val - 0)) % 128))) % 1024, Nat.mod_lt _ (by decide)⟩ else 0)
    let vadd_10 : Fin 8 → Fin 256 → Fin 512 → ℝ := fun i0 i1 i2 => (vslice_backward_4 i0 i1 i2 + vslice_backward_5 i0 i1 i2)
    vadd_10 ⟨(((((i0.val * 512) + i1.val) / 131072) % 8)) % 8, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 256)) % 256, Nat.mod_lt _ (by decide)⟩ i1)

end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_absorb_vs_nonabsorb_wgrad
