import bench.tritonbench_g.rope_embedding.Proofs.SourceBinding
import VeriTile.Triton.Float.ProfiledRewrite

/-! Complete typed source views for the existing index-commutation pair.
The source precision, both rotation directions, every dynamic loop iteration,
both Q stores and the unrestricted input domain are identical on both sides. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding.PairIO
open VeriTile.Triton FP.ProfiledRewrite

def original (p : Source.Parameters) (back : Bool) : Program :=
  ⟨Source.source p back, []⟩

def indexCommuted (p : Source.Parameters) (back : Bool) : Program :=
  ⟨Kernels.indexCommuted p.input p.cosine p.sine p.rowStride p.cosineStride p.sineStride
    p.length p.dimension p.heads p.width back, []⟩

def assumptions : VeriTile.Spec.Assumptions FP.GuardedFragment := []

end VeriTile.Bench.Repairs.RopeEmbedding.PairIO
