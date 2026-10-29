import bench.repairs.tritonbench_g.rope_embedding.IO
import bench.tritonbench_g.fast_rope_embedding.Proofs.FP

namespace VeriTile.Bench.Repairs.RopeEmbedding.PairProof
open VeriTile.Triton PairIO
open scoped VeriTile.Spec

/-- Compare both actual complete bodies directly. Only the natural head-group
product is commuted; both stores, all iterations and directions remain. -/
theorem equivalent (p : Source.Parameters) (back : Bool) :
    original p back ≡[assumptions] indexCommuted p back := by
  cases back <;> equiv_decompose <;> fp_prove

#axiomsClean equivalent
end VeriTile.Bench.Repairs.RopeEmbedding.PairProof
