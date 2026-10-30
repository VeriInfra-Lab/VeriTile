import bench.tritonbench_g.chunk_linear_attn.Proofs.IOResults
import bench.tritonbench_g.chunk_linear_attn.Proofs.IOSafety

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.PublicIO
open VeriTile Triton FP.Structural
noncomputable section

theorem real (c : KernelInterface.Config) (valid : Host.Valid c.dims)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work) :
    (IO.real c work).Implements (Mathematical.real c) := by
  apply ReductionProgramIO.ImplementsOn.intro
  · intro vals _ bounds backed before _
    exact IOSafety.stages Real.algebra Real.dotPrimitive c vals work legal c.scale bounds backed before
  · intro vals _ before binding
    exact IOResults.real c valid work legal vals before binding

#axiomsClean real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.PublicIO
