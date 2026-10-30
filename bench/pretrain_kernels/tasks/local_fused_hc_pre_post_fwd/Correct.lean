import bench.pretrain_kernels.tasks.local_fused_hc_pre_post_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_hc_pre_post_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1,i2) =>
    let input0 := xs.input0
    let input1 := xs.input1
    let input2 := xs.input2
    let input3 := xs.input3
    let vto_copy : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨(((((i0.val * 33554432) + (i1.val * 8192) + i2.val) / 2048) % 4)) % 4, Nat.mod_lt _ (by decide)⟩ ⟨((((i0.val * 33554432) + (i1.val * 8192) + i2.val) % 2048)) % 2048, Nat.mod_lt _ (by decide)⟩
    let vpow_1 : Fin 2 → Fin 4096 → Fin 8192 → ℝ := fun i0 i1 i2 => ((vto_copy i0 i1 i2) * (vto_copy i0 i1 i2))
    let vmean : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => ((∑ lane : Fin 8192, vpow_1 i0 i1 lane) / 8192)
    let vadd : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (vmean i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ + (1/1000000))
    let vrsqrt : Fin 2 → Fin 4096 → Fin 1 → ℝ := fun i0 i1 i2 => (1 / Real.sqrt (vadd i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩))
    let vmm : Fin 8192 → Fin 8 → ℝ := fun i0 i1 => (∑ lane : Fin 8192, vto_copy ⟨(((((i0.val * 8192) + lane.val) / 33554432) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((i0.val % 4096)) % 4096, Nat.mod_lt _ (by decide)⟩ lane * input1 i1 lane)
    let vmul : Fin 2 → Fin 4096 → Fin 8 → ℝ := fun i0 i1 i2 => (vmm ⟨(((i0.val * 4096) + i1.val)) % 8192, Nat.mod_lt _ (by decide)⟩ i2 * vrsqrt i0 i1 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    let vmul_2 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul i0 i1 ⟨((i2.val + 4)) % 8, Nat.mod_lt _ (by decide)⟩ * input2 ⟨(1) % 2, Nat.mod_lt _ (by decide)⟩)
    let vadd_3 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul_2 i0 i1 i2 + input3 ⟨((i2.val + 4)) % 8, Nat.mod_lt _ (by decide)⟩)
    let vsigmoid_1 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (1 / (1 + Real.exp (-(vadd_3 i0 i1 i2))))
    let vmul_3 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vsigmoid_1 i0 i1 i2 * (2 : ℝ))
    let vadd_4 : Fin 2 → Fin 4096 → Fin 4 → ℝ := fun i0 i1 i2 => (vmul_3 i0 i1 i2 + (1/1000000))
    vadd_4 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_fused_hc_pre_post_fwd
