import bench.tritonbench_g.chunk_linear_attn.Proofs.IORun
import bench.tritonbench_g.chunk_linear_attn.Proofs.IOReal
import bench.tritonbench_g.chunk_linear_attn.Proofs.IOFrame

namespace VeriTile.Bench.Repairs.ChunkLinearAttn.IOResults
open VeriTile Triton FP.Structural
noncomputable section

theorem real (c : KernelInterface.Config) (valid : Host.Valid c.dims)
    (work : KernelSchedule.Work) (legal : KernelSchedule.Legal c.call work)
    (vals : (IO.real c work).Inputs) (before : State ℝ)
    (binding : (IO.real c work).InputBinding vals before) :
    ∃ after,runReductionStages ((IO.real c work).stages vals) before = some after ∧
      (IO.real c work).OutputBinding vals (Mathematical.real c vals) after ∧
      (IO.real c work).Frame vals before after := by
  obtain ⟨after,ran,mem⟩ := IORun.stages Real.algebra Real.dotPrimitive c.call work legal before
  have result : KernelExecution.RealResult c.call before.mem after.mem := by
    rw [mem]
    exact KernelExecution.real_model c.call valid before.mem
  refine ⟨after,ran,IOReal.outputs c vals before binding after result,?_⟩
  intro r a outside
  rw [mem]
  exact KernelInterface.complete_frame Real.algebra Real.dotPrimitive c.call before.mem r a
    (IOFrame.outside c vals r a outside)

#axiomsClean real
end
end VeriTile.Bench.Repairs.ChunkLinearAttn.IOResults
