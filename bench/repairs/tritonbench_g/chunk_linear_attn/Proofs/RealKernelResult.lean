import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.RealKernelResult
open VeriTile Triton FP.Structural
noncomputable section

/-- All four complete source grids under legal schedules, the independent
causal formula and full VJP, optional endpoints, and every untouched cell. -/
theorem kernel_correct (c : KernelInterface.Config) (valid : Host.Valid c.dims)
    (before : Memory ℝ) (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    ∃ after,KernelSchedule.run Real.algebra Real.dotPrimitive c.call work before=some after ∧
      KernelExecution.RealResult c.call before after ∧
      (∀ region offset,KernelInterface.Outside c.call region offset → after region offset=before region offset) := by
  rw [KernelSchedule.run_eq Real.algebra Real.dotPrimitive c.call work legal]
  refine ⟨c.call.result before,KernelExecution.source_complete _ _ _ _,
    KernelExecution.real_model c.call valid before,?_⟩
  exact KernelInterface.complete_frame Real.algebra Real.dotPrimitive c.call before

#axiomsClean kernel_correct
#specNonCircular KernelExecution.RealResult avoiding [KernelSchedule.run,KernelExecution.run,FP.Structural.exec]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.RealKernelResult
