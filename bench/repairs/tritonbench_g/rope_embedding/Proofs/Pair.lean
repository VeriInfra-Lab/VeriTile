import bench.repairs.tritonbench_g.rope_embedding.IO
import bench.tritonbench_g.fast_rope_embedding.Proofs.FP

namespace VeriTile.Bench.Repairs.RopeEmbedding.PairProof
open VeriTile.Triton PairIO
open scoped VeriTile.Spec

/-- Reuse the complete typed source proof through the existing exact source
binding. The sole change is Nat multiplication; no FP rewrite is introduced. -/
theorem equivalent (p : Source.Parameters) (back : Bool) :
    original p back ≡[assumptions] indexCommuted p back := by
  exact VeriTile.Bench.TritonBenchG.FastRopeEmbedding.FPProof.equivalent
    .fp32 .fp32 .fp32 p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride
    p.length p.dimension p.heads p.width back

#axiomsClean equivalent
end VeriTile.Bench.Repairs.RopeEmbedding.PairProof
