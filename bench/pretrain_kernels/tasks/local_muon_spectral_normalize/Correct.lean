import bench.pretrain_kernels.tasks.local_muon_spectral_normalize.Contract

namespace VeriTile.Bench.PretrainKernels.local_muon_spectral_normalize
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
    let vlinalg_vector_norm : Fin 3 → Fin 1 → Fin 1 → ℝ := fun i0 i1 i2 => Real.sqrt (∑ lane : Fin 128733, (input0 i0 ⟨(((lane.val / 517) % 249)) % 249, Nat.mod_lt _ (by decide)⟩ ⟨((lane.val % 517)) % 517, Nat.mod_lt _ (by decide)⟩)^2)
    let vclamp : Fin 3 → Fin 1 → Fin 1 → ℝ := fun i0 i1 i2 => max (vlinalg_vector_norm i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩) ((1/10000000))
    let vdiv : Fin 3 → Fin 249 → Fin 517 → ℝ := fun i0 i1 i2 => (input0 i0 i1 i2 / vclamp i0 ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩ ⟨(0) % 1, Nat.mod_lt _ (by decide)⟩)
    vdiv i0 i1 i2)

end
end VeriTile.Bench.PretrainKernels.local_muon_spectral_normalize
