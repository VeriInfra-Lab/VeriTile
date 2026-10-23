import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.RealResult
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Coverage

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
open Checked Binding Host
noncomputable section

/-- Complete arbitrary-Real forward/backward wrapper correctness, including
original input views, returned objects, saved tensors, gradients and frames. -/
theorem chunk_gate_recurrence_correct {M β : Type} (api : API M ℝ β) (c : Shape) (seed : Bool)
    (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true)
    (tile : Reduction.ProductTree 1024) (host : Reduction.Tree c.tiles)
    (plans : planCheck c tile host=Bool.true) (call : Invocation api realOps c seed runtime) :
    Public.RealClaim api c seed runtime (trees c tile host plans) call
      (forward api realOps c seed runtime call.forwardPrepared)
      (backward api realOps c seed runtime (trees c tile host plans) call.backwardPrepared) :=
  Public.real_complete api c seed runtime accepted _ call

#axiomsClean chunk_gate_recurrence_correct
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
