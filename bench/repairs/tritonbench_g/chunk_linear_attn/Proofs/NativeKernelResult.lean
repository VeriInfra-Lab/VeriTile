import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.KernelInterface

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.NativeKernelResult
open VeriTile Triton FP.Structural
noncomputable section

/-- Complete four-source native execution on raw input words and the actual
scale ABI word. The independent coordinate target and full exterior frame use
the unchanged scalar/FMA profile and finite reduction trees. -/
theorem kernel_native (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (valid : Host.Valid c.dims) (scale : BitVec 32) (before : RegionName → Nat → BitVec 32)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    ∃ after,KernelSchedule.runNative profile c.call scale work before=some after ∧
      after=KernelExecution.nativeTarget profile c.call valid scale before ∧
      (∀ region offset,KernelInterface.Outside c.call region offset → after region offset=before region offset) := by
  rw [KernelSchedule.native_eq profile c.call scale work legal]
  exact ⟨_,KernelExecution.native_complete profile c.call valid scale before,rfl,
    KernelInterface.native_frame profile c.call valid scale before⟩

#axiomsClean kernel_native
#specNonCircular KernelExecution.nativeTarget avoiding [KernelExecution.run,KernelExecution.runNative,
  FP.Structural.exec,runGrid,executeFour,gridMemory,stageMemory,modelFour,completeMemory,
  NativeTerms.rawMemory,NativeBound.rawMemory]
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.NativeKernelResult
