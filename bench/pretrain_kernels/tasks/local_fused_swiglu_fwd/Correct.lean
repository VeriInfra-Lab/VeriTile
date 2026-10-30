import bench.pretrain_kernels.tasks.local_fused_swiglu_fwd.Contract

namespace VeriTile.Bench.PretrainKernels.local_fused_swiglu_fwd
open VeriTile Triton
noncomputable section
set_option maxRecDepth 50000
set_option maxHeartbeats 4000000
set_option linter.unusedTactic false
open scoped BigOperators Classical VeriTile.Triton.LogicalScheduledKernelIO

/-- Unsolved complete Real execution, safety, output and exterior-frame goal. -/
def correctnessProblem : Prop :=
  Spec.Real (IO.realIO ⊨ fun xs (i0,i1) =>
    let input0 := xs.input0
    let vto_copy : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => input0 i0 ⟨(i1.val) % 1280, Nat.mod_lt _ (by decide)⟩
    let vsilu : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vto_copy i0 i1 * (1 / (1 + Real.exp (-(vto_copy i0 i1)))))
    let vto_copy_1 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => input0 i0 ⟨((i1.val + 640)) % 1280, Nat.mod_lt _ (by decide)⟩
    let vmul : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => (vsilu i0 i1 * vto_copy_1 i0 i1)
    let vto_copy_2 : Fin 4096 → Fin 640 → ℝ := fun i0 i1 => vmul i0 i1
    vto_copy_2 i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_fused_swiglu_fwd
