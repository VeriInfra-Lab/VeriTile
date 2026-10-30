import bench.repairs.tritonbench_g.rope_embedding.Proofs.Source
import bench.tritonbench_g.fast_rope_embedding.Kernels
import VeriTile.Meta.StatementAudit

namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton

/-- The local readable kernel has the exact typed AST used by the retained
full-loop proof: FP32 Q/cos/sin, both directions and every statement. -/
theorem selected_source_binding (p : Source.Parameters) (back : Bool) :
    Source.source p back=
      VeriTile.Bench.TritonBenchG.FastRopeEmbedding.TypedSource.rope .fp32 .fp32 .fp32 Bool.false
        p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride
        p.length p.dimension p.heads p.width back := rfl

#axiomsClean selected_source_binding
end VeriTile.Bench.Repairs.RopeEmbedding
