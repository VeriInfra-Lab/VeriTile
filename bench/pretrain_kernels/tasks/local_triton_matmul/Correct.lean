import bench.pretrain_kernels.tasks.local_triton_matmul.Contract

namespace VeriTile.Bench.PretrainKernels.local_triton_matmul
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
    let input0 := xs.input0
    let vmul : Fin 65 → Fin 257 → ℝ := fun i0 i1 => (parameter0 i0 i1 * (1/50))
    let vmul_1 : Fin 65 → ℝ := fun i0 => (parameter1 i0 * (1/50))
    let vaddmm : Fin 65 → Fin 65 → ℝ := fun i0 i1 => (∑ lane : Fin 257, input0 i0 lane * vmul i1 lane) + vmul_1 i1 * 1
    vaddmm i0 i1)

end
end VeriTile.Bench.PretrainKernels.local_triton_matmul
