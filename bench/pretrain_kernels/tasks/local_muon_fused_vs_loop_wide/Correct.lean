import bench.pretrain_kernels.tasks.local_muon_fused_vs_loop_wide.Contract

namespace VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop_wide
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
    let input0 := xs.input0
    let vlerp : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (0 + (input0 ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - 0) * (12500000000000011/250000000000000000))
    let vselect_scatter : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 0) then vlerp i1 i2 else 0)
    let vlerp_1 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (input0 ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vselect_scatter ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - input0 ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1) * (19/20))
    let vlinalg_vector_norm : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 35904, (vlerp_1 ⟨(((lane.val / 264) % 136)) % 136, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 264)) % 264, Nat.mod_lt _ (by decide)⟩)^2)
    let vclamp : Fin 1 → ℝ := fun i0 => max (vlinalg_vector_norm ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) ((1/10000000))
    let vdiv : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vlerp_1 i0 i1 / vclamp ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmm : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vdiv i0 lane * vdiv i1 lane)
    let vaddmm : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm i0 lane * vmm lane i1) * (4063/2000)) + vmm i0 i1 * (-191/40)
    let vaddmm_1 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm i0 lane * vdiv lane i1) + vdiv i0 i1 * (6889/2000)
    let vmm_1 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_1 i0 lane * vaddmm_1 i1 lane)
    let vaddmm_2 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_1 i0 lane * vmm_1 lane i1) * (4063/2000)) + vmm_1 i0 i1 * (-191/40)
    let vaddmm_3 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_2 i0 lane * vaddmm_1 lane i1) + vaddmm_1 i0 i1 * (6889/2000)
    let vmm_2 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_3 i0 lane * vaddmm_3 i1 lane)
    let vaddmm_4 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_2 i0 lane * vmm_2 lane i1) * (4063/2000)) + vmm_2 i0 i1 * (-191/40)
    let vaddmm_5 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_4 i0 lane * vaddmm_3 lane i1) + vaddmm_3 i0 i1 * (6889/2000)
    let vmm_3 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_5 i0 lane * vaddmm_5 i1 lane)
    let vaddmm_6 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_3 i0 lane * vmm_3 lane i1) * (4063/2000)) + vmm_3 i0 i1 * (-191/40)
    let vaddmm_7 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_6 i0 lane * vaddmm_5 lane i1) + vaddmm_5 i0 i1 * (6889/2000)
    let vmm_4 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_7 i0 lane * vaddmm_7 i1 lane)
    let vaddmm_8 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_4 i0 lane * vmm_4 lane i1) * (4063/2000)) + vmm_4 i0 i1 * (-191/40)
    let vaddmm_9 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_8 i0 lane * vaddmm_7 lane i1) + vaddmm_7 i0 i1 * (6889/2000)
    let vmul : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (parameter0 ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 * (9999/10000))
    let vselect_scatter_1 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 0) then vmul i1 i2 else parameter0 i0 i1 i2)
    let vadd : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_1 ⟨(0) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vaddmm_9 i0 i1 * (-649923072370877/20000000000000000)))
    let vselect_scatter_2 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 0) then vadd i1 i2 else vselect_scatter_1 i0 i1 i2)
    let vlerp_2 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (input0 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - vselect_scatter ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1) * (12500000000000011/250000000000000000))
    let vselect_scatter_3 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 1) then vlerp_2 i1 i2 else vselect_scatter i0 i1 i2)
    let vlerp_3 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (input0 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vselect_scatter_3 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - input0 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1) * (19/20))
    let vlinalg_vector_norm_1 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 35904, (vlerp_3 ⟨(((lane.val / 264) % 136)) % 136, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 264)) % 264, Nat.mod_lt _ (by decide)⟩)^2)
    let vclamp_1 : Fin 1 → ℝ := fun i0 => max (vlinalg_vector_norm_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) ((1/10000000))
    let vdiv_1 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vlerp_3 i0 i1 / vclamp_1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmm_5 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vdiv_1 i0 lane * vdiv_1 i1 lane)
    let vaddmm_10 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_5 i0 lane * vmm_5 lane i1) * (4063/2000)) + vmm_5 i0 i1 * (-191/40)
    let vaddmm_11 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_10 i0 lane * vdiv_1 lane i1) + vdiv_1 i0 i1 * (6889/2000)
    let vmm_6 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_11 i0 lane * vaddmm_11 i1 lane)
    let vaddmm_12 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_6 i0 lane * vmm_6 lane i1) * (4063/2000)) + vmm_6 i0 i1 * (-191/40)
    let vaddmm_13 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_12 i0 lane * vaddmm_11 lane i1) + vaddmm_11 i0 i1 * (6889/2000)
    let vmm_7 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_13 i0 lane * vaddmm_13 i1 lane)
    let vaddmm_14 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_7 i0 lane * vmm_7 lane i1) * (4063/2000)) + vmm_7 i0 i1 * (-191/40)
    let vaddmm_15 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_14 i0 lane * vaddmm_13 lane i1) + vaddmm_13 i0 i1 * (6889/2000)
    let vmm_8 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_15 i0 lane * vaddmm_15 i1 lane)
    let vaddmm_16 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_8 i0 lane * vmm_8 lane i1) * (4063/2000)) + vmm_8 i0 i1 * (-191/40)
    let vaddmm_17 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_16 i0 lane * vaddmm_15 lane i1) + vaddmm_15 i0 i1 * (6889/2000)
    let vmm_9 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_17 i0 lane * vaddmm_17 i1 lane)
    let vaddmm_18 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_9 i0 lane * vmm_9 lane i1) * (4063/2000)) + vmm_9 i0 i1 * (-191/40)
    let vaddmm_19 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_18 i0 lane * vaddmm_17 lane i1) + vaddmm_17 i0 i1 * (6889/2000)
    let vmul_1 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_2 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 * (9999/10000))
    let vselect_scatter_4 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 1) then vmul_1 i1 i2 else vselect_scatter_2 i0 i1 i2)
    let vadd_1 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_4 ⟨(1) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vaddmm_19 i0 i1 * (-649923072370877/20000000000000000)))
    let vselect_scatter_5 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 1) then vadd_1 i1 i2 else vselect_scatter_4 i0 i1 i2)
    let vlerp_4 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_3 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (input0 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - vselect_scatter_3 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1) * (12500000000000011/250000000000000000))
    let vselect_scatter_6 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 2) then vlerp_4 i1 i2 else vselect_scatter_3 i0 i1 i2)
    let vlerp_5 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (input0 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vselect_scatter_6 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 - input0 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1) * (19/20))
    let vlinalg_vector_norm_2 : Fin 1 → ℝ := fun i0 => Real.sqrt (∑ lane : Fin 35904, (vlerp_5 ⟨(((lane.val / 264) % 136)) % 136, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 264)) % 264, Nat.mod_lt _ (by decide)⟩)^2)
    let vclamp_2 : Fin 1 → ℝ := fun i0 => max (vlinalg_vector_norm_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) ((1/10000000))
    let vdiv_2 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vlerp_5 i0 i1 / vclamp_2 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmm_10 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vdiv_2 i0 lane * vdiv_2 i1 lane)
    let vaddmm_20 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_10 i0 lane * vmm_10 lane i1) * (4063/2000)) + vmm_10 i0 i1 * (-191/40)
    let vaddmm_21 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_20 i0 lane * vdiv_2 lane i1) + vdiv_2 i0 i1 * (6889/2000)
    let vmm_11 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_21 i0 lane * vaddmm_21 i1 lane)
    let vaddmm_22 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_11 i0 lane * vmm_11 lane i1) * (4063/2000)) + vmm_11 i0 i1 * (-191/40)
    let vaddmm_23 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_22 i0 lane * vaddmm_21 lane i1) + vaddmm_21 i0 i1 * (6889/2000)
    let vmm_12 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_23 i0 lane * vaddmm_23 i1 lane)
    let vaddmm_24 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_12 i0 lane * vmm_12 lane i1) * (4063/2000)) + vmm_12 i0 i1 * (-191/40)
    let vaddmm_25 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_24 i0 lane * vaddmm_23 lane i1) + vaddmm_23 i0 i1 * (6889/2000)
    let vmm_13 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_25 i0 lane * vaddmm_25 i1 lane)
    let vaddmm_26 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_13 i0 lane * vmm_13 lane i1) * (4063/2000)) + vmm_13 i0 i1 * (-191/40)
    let vaddmm_27 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_26 i0 lane * vaddmm_25 lane i1) + vaddmm_25 i0 i1 * (6889/2000)
    let vmm_14 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => (∑ lane : Fin 264, vaddmm_27 i0 lane * vaddmm_27 i1 lane)
    let vaddmm_28 : Fin 136 → Fin 136 → ℝ := fun i0 i1 => ((∑ lane : Fin 136, vmm_14 i0 lane * vmm_14 lane i1) * (4063/2000)) + vmm_14 i0 i1 * (-191/40)
    let vaddmm_29 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (∑ lane : Fin 136, vaddmm_28 i0 lane * vaddmm_27 lane i1) + vaddmm_27 i0 i1 * (6889/2000)
    let vmul_2 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_5 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 * (9999/10000))
    let vselect_scatter_7 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 2) then vmul_2 i1 i2 else vselect_scatter_5 i0 i1 i2)
    let vadd_2 : Fin 136 → Fin 264 → ℝ := fun i0 i1 => (vselect_scatter_7 ⟨(2) % 3, Nat.mod_lt _ (by decide)⟩ i0 i1 + (vaddmm_29 i0 i1 * (-649923072370877/20000000000000000)))
    let vselect_scatter_8 : Fin 3 → Fin 136 → Fin 264 → ℝ := fun i0 i1 i2 => (if (i0.val = 2) then vadd_2 i1 i2 else vselect_scatter_7 i0 i1 i2)
    vselect_scatter_8 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_muon_fused_vs_loop_wide
