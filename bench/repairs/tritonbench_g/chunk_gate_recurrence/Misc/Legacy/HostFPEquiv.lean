import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.FPResult
import bench.repairs.tritonbench_g.chunk_gate_recurrence.Proofs.Coverage

namespace VeriTile.Bench.Repairs.ChunkGateRecurrence
open Checked Binding Host
noncomputable section

/-- Complete raw-word/byte wrapper result. Only the three named local scalar
laws remain numerical conditions; all structural and source obligations close. -/
theorem chunk_gate_recurrence_fp32 {M : Type} (api : API M Word Public.Bytes.Byte)
    (codec : Public.Bytes.Codec api) (isa : ISA) (laws : LocalConformance isa allOperands)
    (c : Shape) (seed : Bool) (runtime : Runtime c seed) (accepted : check c seed runtime=Bool.true)
    (tile : Reduction.ProductTree 1024) (host : Reduction.Tree c.tiles)
    (plans : planCheck c tile host=Bool.true) (call : Invocation api (wordOps isa) c seed runtime) :
    Public.FPClaim api isa c seed runtime (trees c tile host plans) call
      (forward api (wordOps isa) c seed runtime call.forwardPrepared)
      (nativeBackward api isa c seed runtime (trees c tile host plans) call.backwardPrepared) :=
  Public.fp_complete api codec isa laws c seed runtime accepted _ call

#axiomsClean chunk_gate_recurrence_fp32
end
end VeriTile.Bench.Repairs.ChunkGateRecurrence
