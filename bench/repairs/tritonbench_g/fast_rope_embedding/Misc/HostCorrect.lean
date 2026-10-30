import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Backward

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Bench.Repairs
/-- Full sequential Q/K forward, original returned views and caches, and final
shared-memory values/frame after both source calls. -/
theorem fast_rope_forward_correct (api : RopeEmbedding.Memory.API ℝ Store Byte)
    (rq rk : RopeEmbedding.Request) (before : Store) (p : PairPreparation api id rq rk before)
    (laws : StorageLaws api id rq rk before p)
    (accepted : check api id id RopeEmbedding.ideal rq rk before p=Bool.true) : RealOutcome api rq rk before p :=
  whole_real api rq rk before p laws accepted

/-- Actual callback triple and gradient view, using current cached coefficients.
A fixed-forward adjoint interpretation additionally requires unchanged coefficients. -/
theorem fast_rope_backward_correct (api : RopeEmbedding.Memory.API ℝ Store Byte)
    (r : RopeEmbedding.Request) (before : Store) (p : BackwardPreparation api id r before)
    (accepted : backwardCheck api id r before p=Bool.true) : BackwardRealOutcome api r before p :=
  whole_backward_real api r before p accepted
#axiomsClean fast_rope_forward_correct
#axiomsClean fast_rope_backward_correct
end VeriTile.Bench.Repairs.FastRopeEmbedding
