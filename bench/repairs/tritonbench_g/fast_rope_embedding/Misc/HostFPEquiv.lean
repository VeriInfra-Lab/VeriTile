import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Backward

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Bench.Repairs
/-- Native words and four-byte outputs in final shared memory after both calls;
source execution, cached objects, ownership and caller frame are proved. -/
theorem fast_rope_forward_fp32
    (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8))
    (n : RopeEmbedding.Numerics.Primitives) (rq rk : RopeEmbedding.Request) (before : Store)
    (p : PairPreparation api RopeEmbedding.Numerics.leaf rq rk before)
    (laws : StorageLaws api RopeEmbedding.Numerics.leaf rq rk before p)
    (codec : ∀ w k,api.encode w k=RopeEmbedding.Numerics.wordByte w k)
    (accepted : check api RopeEmbedding.Numerics.leaf RopeEmbedding.Numerics.Tagged.word
      (RopeEmbedding.Numerics.operations n) rq rk before p=Bool.true) : FP32Outcome api n rq rk before p :=
  whole_fp32 api n rq rk before p laws codec accepted

/-- Native backward on the current plain ctx coefficient objects, including the
exact (gradient, None, None) result and original gradient reshape reuse. -/
theorem fast_rope_backward_fp32
    (api : RopeEmbedding.Memory.API RopeEmbedding.Numerics.Word Store (BitVec 8))
    (n : RopeEmbedding.Numerics.Primitives) (r : RopeEmbedding.Request) (before : Store)
    (p : BackwardPreparation api RopeEmbedding.Numerics.leaf r before)
    (codec : ∀ w k,api.encode w k=RopeEmbedding.Numerics.wordByte w k)
    (accepted : backwardCheck api RopeEmbedding.Numerics.leaf r before p=Bool.true) :
    BackwardFP32Outcome api n r before p := whole_backward_fp32 api n r before p codec accepted
#axiomsClean fast_rope_forward_fp32
#axiomsClean fast_rope_backward_fp32
end VeriTile.Bench.Repairs.FastRopeEmbedding
