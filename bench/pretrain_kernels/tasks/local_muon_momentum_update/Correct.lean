import bench.pretrain_kernels.tasks.local_muon_momentum_update.Contract

namespace VeriTile.Bench.PretrainKernels.local_muon_momentum_update
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
    let vmul : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (parameter0 i0 i1 i2 * (1/10))
    let vlerp : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (vmul i0 i1 i2 + (input0 i0 i1 i2 - vmul i0 i1 i2) * (12500000000000011/250000000000000000))
    let vlerp_1 : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (input0 i0 i1 i2 + (vlerp i0 i1 i2 - input0 i0 i1 i2) * (19/20))
    vlerp_1 i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_muon_momentum_update
