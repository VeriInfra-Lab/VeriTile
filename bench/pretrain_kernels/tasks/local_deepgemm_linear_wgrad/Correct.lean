import bench.pretrain_kernels.tasks.local_deepgemm_linear_wgrad.Contract

namespace VeriTile.Bench.PretrainKernels.local_deepgemm_linear_wgrad
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
    let input1 := xs.input1
    let vmm_1 : Fin 2048 → Fin 2048 → ℝ := fun i0 i1 => (∑ lane : Fin 256, input1 ⟨(((((lane.val * 2048) + i0.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i0 * input0 ⟨(((((lane.val * 2048) + i1.val) / 262144) % 2)) % 2, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 128)) % 128, Nat.mod_lt _ (by decide)⟩ i1)
    vmm_1 i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_deepgemm_linear_wgrad
