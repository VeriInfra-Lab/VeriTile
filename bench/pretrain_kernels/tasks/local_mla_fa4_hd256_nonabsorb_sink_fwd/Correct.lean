import bench.pretrain_kernels.tasks.local_mla_fa4_hd256_nonabsorb_sink_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_sink_fwd
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
    let parameter7 := xs.parameter7
    let input0 := xs.input0
    let vmul : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_1 : Fin 256 → ℝ := fun i0 => (parameter1 i0 * (1/10))
    let vadd : Fin 256 → ℝ := fun i0 => (vmul_1 i0 + 1)
    let vmul_2 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => (parameter2 i0 i1 * (1/16))
    let vmul_3 : Fin 320 → Fin 512 → ℝ := fun i0 i1 => (parameter3 i0 i1 * (2209708691207961/50000000000000000))
    let vmul_4 : Fin 256 → ℝ := fun i0 => (parameter4 i0 * (1/10))
    let vadd_1 : Fin 256 → ℝ := fun i0 => (vmul_4 i0 + 1)
    let vmul_5 : Fin 3584 → Fin 256 → ℝ := fun i0 i1 => (parameter5 i0 i1 * (1/16))
    let vmul_6 : Fin 512 → Fin 2048 → ℝ := fun i0 i1 => (parameter6 i0 i1 * (2209708691207961/100000000000000000))
    let vto_copy : Fin 256 → Fin 512 → ℝ := fun i0 i1 => vmul i0 i1
    let vto_copy_1 : Fin 256 → ℝ := fun i0 => vadd i0
    let vto_copy_2 : Fin 2048 → Fin 256 → ℝ := fun i0 i1 => vmul_2 i0 i1
    let vto_copy_3 : Fin 320 → Fin 512 → ℝ := fun i0 i1 => vmul_3 i0 i1
    let vto_copy_4 : Fin 256 → ℝ := fun i0 => vadd_1 i0
    let vto_copy_5 : Fin 3584 → Fin 256 → ℝ := fun i0 i1 => vmul_5 i0 i1
    let vto_copy_6 : Fin 512 → Fin 2048 → ℝ := fun i0 i1 => vmul_6 i0 i1
    let vmul_7 : Fin 8 → ℝ := fun i0 => (parameter7 i0 * (1/2))
    let vto_copy_14 : Fin 8 → ℝ := fun i0 => vmul_7 i0
    let vtriu : Fin 128 → Fin 128 → Bool := fun i0 i1 => (if i1.val ≥ i0.val + 1 then Bool.true else Bool.false)
    let vmasked_fill : Fin 2 → Fin 1 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (if vtriu i2 i3 then (-338953138925153547590470800371487866880 : ℝ) else 0)
    let vmm : Fin 256 → Fin 256 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy i1 lane)
    let vto_copy_18 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2
    let vpow_1 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_18 i0 i1 i2) * (vto_copy_18 i0 i1 i2))
    let vmean : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_1 i0 i1 lane) / 256)
    let vadd_2 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_2 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_8 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_18 i0 i1 i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_9 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_1 i2 * vmul_8 i0 i1 i2)
    let vto_copy_19 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_9 i0 i1 i2
    let vmm_1 : Fin 256 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_19 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_2 i1 lane)
    let vmm_2 : Fin 256 → Fin 320 → ℝ := fun i0 i1 => (∑ lane : Fin 512, input0 ⟨(((((i0.val * 512) + lane.val) / 65536) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_3 i1 lane)
    let vto_copy_20 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmm_2 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(i2.val) % 320, Nat.mod_lt _ (by decide)⟩
    let vpow_2 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => ((vto_copy_20 i0 i1 i2) * (vto_copy_20 i0 i1 i2))
    let vmean_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 256, vpow_2 i0 i1 lane) / 256)
    let vadd_3 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt_1 : Fin 2 → Fin 128 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd_3 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmul_10 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_20 i0 i1 i2 * vrsqrt_1 i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_11 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (vto_copy_4 i2 * vmul_10 i0 i1 i2)
    let vto_copy_21 : Fin 2 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => vmul_11 i0 i1 i2
    let vmm_3 : Fin 256 → Fin 3584 → ℝ := fun i0 i1 => (∑ lane : Fin 256, vto_copy_21 ⟨(((((i0.val * 256) + lane.val) / 32768) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ lane * vto_copy_5 i1 lane)
    let vmul_12 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 256) + (i3.val * 2) + 192)) % 2048, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_13 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 256) + (i3.val * 2) + 193)) % 2048, Nat.mod_lt _ (by decide)⟩ * 0)
    let vsub : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_12 i0 i1 i2 i3 - vmul_13 i0 i1 i2 i3)
    let vmul_14 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 256) + (i3.val * 2) + 193)) % 2048, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_15 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_1 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i1.val * 256) + (i3.val * 2) + 192)) % 2048, Nat.mod_lt _ (by decide)⟩ * 0)
    let vadd_4 : Fin 2 → Fin 8 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_14 i0 i1 i2 i3 + vmul_15 i0 i1 i2 i3)
    let vcat : Fin 2 → Fin 8 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_4 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vmul_16 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_17 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * 0)
    let vsub_1 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_16 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 - vmul_17 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vmul_18 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 257)) % 320, Nat.mod_lt _ (by decide)⟩ * 1)
    let vmul_19 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmm_2 ⟨(((i0.val * 128) + i2.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨(((i3.val * 2) + 256)) % 320, Nat.mod_lt _ (by decide)⟩ * 0)
    let vadd_5 : Fin 2 → Fin 1 → Fin 128 → Fin 32 → ℝ := fun i0 i1 i2 i3 => (vmul_18 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3 + vmul_19 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_1 : Fin 2 → Fin 1 → Fin 128 → Fin 64 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 32 then vsub_1 i0 i1 i2 ⟨((i3.val - 0)) % 32, Nat.mod_lt _ (by decide)⟩ else vadd_5 i0 i1 i2 ⟨((i3.val - 32)) % 32, Nat.mod_lt _ (by decide)⟩)
    let vcat_2 : Fin 2 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 192 then vmm_1 ⟨(((((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 262144) % 2) * 128) + ((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) / 2048) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 262144) + (i2.val * 2048) + (i1.val * 256) + (i3.val - 0)) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩ else vcat i0 i1 i2 ⟨((i3.val - 192)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vcat_3 : Fin 2 → Fin 8 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 192 then vmm_3 ⟨(((((((i0.val * 458752) + (i2.val * 3584) + (i1.val * 448) + (i3.val - 0)) / 458752) % 2) * 128) + ((((i0.val * 458752) + (i2.val * 3584) + (i1.val * 448) + (i3.val - 0)) / 3584) % 128))) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 458752) + (i2.val * 3584) + (i1.val * 448) + (i3.val - 0)) % 3584)) % 3584, Nat.mod_lt _ (by decide)⟩ else vcat_1 i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 ⟨((i3.val - 192)) % 64, Nat.mod_lt _ (by decide)⟩)
    let vbmm : Fin 16 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 256, vcat_2 ⟨(((((i0.val * 32768) + (i1.val * 256) + lane.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 lane * vcat_3 ⟨(((((i0.val * 32768) + (lane.val * 128) + i2.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i2 lane)
    let vmul_20 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vbmm ⟨(((i0.val * 8) + i1.val)) % 16, Nat.mod_lt _ (by decide)⟩ i2 i3 * (1/16))
    let vadd_6 : Fin 2 → Fin 8 → Fin 128 → Fin 128 → ℝ := fun i0 i1 i2 i3 => (vmul_20 i0 i1 i2 i3 + vmasked_fill i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ i2 i3)
    let vcat_4 : Fin 2 → Fin 8 → Fin 128 → Fin 129 → ℝ := fun i0 i1 i2 i3 => (if i3.val < 128 then vadd_6 i0 i1 i2 ⟨((i3.val - 0)) % 128, Nat.mod_lt _ (by decide)⟩ else vto_copy_14 i1)
    let vto_copy_22 : Fin 2 → Fin 8 → Fin 128 → Fin 129 → ℝ := fun i0 i1 i2 i3 => vcat_4 i0 i1 i2 i3
    let vsoftmax : Fin 2 → Fin 8 → Fin 128 → Fin 129 → ℝ := fun i0 i1 i2 i3 => Real.exp (vto_copy_22 i0 i1 i2 i3) / (∑ lane : Fin 129, Real.exp (vto_copy_22 i0 i1 i2 ⟨(lane.val) % 129, Nat.mod_lt _ (by decide)⟩))
    let vto_copy_23 : Fin 2 → Fin 8 → Fin 128 → Fin 129 → ℝ := fun i0 i1 i2 i3 => vsoftmax i0 i1 i2 i3
    let vbmm_1 : Fin 16 → Fin 128 → Fin 256 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 128, vto_copy_23 ⟨(((((i0.val * 16384) + (i1.val * 128) + lane.val) / 131072) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 8)) % 8, Nat.mod_lt _ (by decide)⟩ i1 ⟨(lane.val) % 129, Nat.mod_lt _ (by decide)⟩ * vmm_3 ⟨(((((((i0.val * 32768) + (lane.val * 256) + i2.val) / 262144) % 2) * 128) + lane.val)) % 256, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val % 8) * 448) + i2.val + 192)) % 3584, Nat.mod_lt _ (by decide)⟩)
    let vmm_4 : Fin 256 → Fin 512 → ℝ := fun i0 i1 => (∑ lane : Fin 2048, vbmm_1 ⟨(((((((i0.val * 2048) + lane.val) / 262144) % 2) * 8) + ((((((((i0.val * 2048) + lane.val) / 262144) % 2) * 262144) + ((i0.val % 128) * 2048) + lane.val) / 256) % 8))) % 16, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ ⟨((((((((i0.val * 2048) + lane.val) / 262144) % 2) * 262144) + ((i0.val % 128) * 2048) + lane.val) % 256)) % 256, Nat.mod_lt _ (by decide)⟩ * vto_copy_6 i1 lane)
    vmm_4 ⟨(((i0.val * 128) + i1.val)) % 256, Nat.mod_lt _ (by decide)⟩ i2)

end
end VeriTile.Bench.PretrainKernels.local_mla_fa4_hd256_nonabsorb_sink_fwd
