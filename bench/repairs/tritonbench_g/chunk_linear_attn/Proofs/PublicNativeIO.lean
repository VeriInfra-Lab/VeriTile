import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IORun
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOSafety
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IONative
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOFrame
import bench.repairs.tritonbench_g.chunk_linear_attn.Proofs.IOLayoutFacts

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PublicNativeIO
open VeriTile Triton FP.Structural
noncomputable section

theorem native (profile : NativeTerms.FP32Profile) (c : KernelInterface.Config)
    (valid : Host.Valid c.dims) (scale : BitVec 32) (work : KernelSchedule.Work)
    (legal : KernelSchedule.Legal c.call work) :
    (IO.native profile c scale work).Implements (NativeFormula.native profile c valid scale) := by
  apply ReductionProgramIO.ImplementsOn.intro
  · intro vals _ bounds backed before _
    exact IOSafety.stages NativeTerms.algebra NativeTerms.dotPrimitive c vals work legal 1 bounds
      (IOLayoutFacts.native_bounds c vals bounds backed) before
  · intro vals _ before binding
    have symbolicLegal : KernelSchedule.Legal (KernelExecution.symbolicCall c.call) work :=
      ⟨legal.forwardState,legal.forwardOutput,legal.backwardState,legal.backwardGradient⟩
    obtain ⟨after,ran,mem⟩ := IORun.stages NativeTerms.algebra NativeTerms.dotPrimitive
      (KernelExecution.symbolicCall c.call) work symbolicLegal before
    refine ⟨after,ran,IONative.outputs profile c valid scale work vals before after binding mem,?_⟩
    intro r a outside
    rw [mem]
    exact KernelInterface.complete_frame NativeTerms.algebra NativeTerms.dotPrimitive
      (KernelExecution.symbolicCall c.call) before.mem r a
      (IOFrame.outside c vals r a (IOLayoutFacts.native_outside c vals r a outside))

#axiomsClean native
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PublicNativeIO
