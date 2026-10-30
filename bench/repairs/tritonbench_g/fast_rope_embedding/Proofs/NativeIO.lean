import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.Interface

namespace VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
noncomputable section

theorem source_pair_tagged (p : Numerics.Primitives) (back second : Bool)
    (a b c s : Numerics.Tagged) :
    scalarPair (Numerics.operations p) back a b c s second =
      Numerics.leaf (Numerics.pair p back second a.word b.word c.word s.word) := by
  cases back <;> cases second <;> rfl

theorem native_implements (cfg : FP.Config) (r : Request)
    (qStride cosineStride sineStride : Nat) (backward : Bool) (row group : Nat)
    (launch : r.Launched row group) (strideEq : row * qStride = row * (r.heads * r.dimension))
    (cosineEq : cosineStride = r.half) (sineEq : sineStride = r.half) :
    (programIO (Numerics.operations (Numerics.softwarePrimitives cfg)) r
      qStride cosineStride sineStride backward row group).Implements
      (fun values => result (fun h c => Numerics.leaf
        (Numerics.pair (Numerics.softwarePrimitives cfg) backward (decide (r.half ≤ c.val))
          (query values h (left r (lane r c))).word (query values h (right r (lane r c))).word
          (cosine values (lane r c)).word (sine values (lane r c)).word))) := by
  simpa only [source_pair_tagged] using implements (Numerics.operations (Numerics.softwarePrimitives cfg))
    r qStride cosineStride sineStride backward row group launch strideEq cosineEq sineEq

#axiomsClean source_pair_tagged
#axiomsClean native_implements
end
end VeriTile.Bench.Repairs.FastRopeEmbedding.Interface
