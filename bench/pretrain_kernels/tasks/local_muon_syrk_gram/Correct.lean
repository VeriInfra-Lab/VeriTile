import bench.pretrain_kernels.tasks.local_muon_syrk_gram.Contract

namespace VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
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
    let vbmm : Fin 3 → Fin 1544 → Fin 1544 → ℝ := fun i0 i1 i2 => (∑ lane : Fin 520, input0 i0 i1 lane * input0 i0 i2 lane)
    vbmm i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_muon_syrk_gram
