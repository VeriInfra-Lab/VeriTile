import bench.pretrain_kernels.tasks.local_situ_glu_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_situ_glu_fwd
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
    let vto_copy : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨(i2.val) % 12288, Nat.mod_lt _ (by decide)⟩
    let vto_copy_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => input0 i0 i1 ⟨((i2.val + 6144)) % 12288, Nat.mod_lt _ (by decide)⟩
    let vdiv : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy i0 i1 i2 / (4 : ℝ))
    let vtanh : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (Real.tanh (vdiv i0 i1 i2))
    let vmul : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh i0 i1 i2 * (4 : ℝ))
    let vsigmoid : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (1 / (1 + Real.exp (-(vto_copy i0 i1 i2))))
    let vmul_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul i0 i1 i2 * vsigmoid i0 i1 i2)
    let vdiv_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vto_copy_1 i0 i1 i2 / (25 : ℝ))
    let vtanh_1 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (Real.tanh (vdiv_1 i0 i1 i2))
    let vmul_2 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vtanh_1 i0 i1 i2 * (25 : ℝ))
    let vmul_3 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => (vmul_1 i0 i1 i2 * vmul_2 i0 i1 i2)
    let vto_copy_2 : Fin 2 → Fin 128 → Fin 6144 → ℝ := fun i0 i1 i2 => vmul_3 i0 i1 i2
    vto_copy_2 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_situ_glu_fwd
