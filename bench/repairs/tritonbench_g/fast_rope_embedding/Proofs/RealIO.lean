import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Interface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

theorem real_implements (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) (launch : r.Launched row group)
    (strideEq : row * qStride = row * (r.heads * r.dimension))
    (cosineEq : cosineStride = r.half) (sineEq : sineStride = r.half) :
    (programIO ideal r qStride cosineStride sineStride backward row group).Implements
      (fun values => result (fun h c =>
        let pair := Spec.pair backward
          (query values h (left r (lane r c))) (query values h (right r (lane r c)))
          (cosine values (lane r c)) (sine values (lane r c))
        if r.half ≤ c.val then pair.2 else pair.1)) := by
  simpa only [ideal_pair, decide_eq_true_eq] using
    implements ideal r qStride cosineStride sineStride backward row group launch strideEq cosineEq sineEq

#axiomsClean real_implements
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
