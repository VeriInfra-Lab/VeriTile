import bench.repairs.tritonbench_g.fast_rope_embedding.Proofs.NativeIO

namespace VeriTile.Bench.Repairs.FastRopeEmbedding
open VeriTile.Triton FP.Structural
open VeriTile.Bench.Repairs.RopeEmbedding
open scoped VeriTile.Triton.FP.Structural.ProgramIO
noncomputable section

/-- The FP32 source and its existing contraction model. Tagged values keep
arbitrary input product markers; every FP32 word remains admitted. The source
loads/casts discard those markers before computing the native pair. -/
def nativeIO (cfg : FP.Config) (r : Request) (qStride cosineStride sineStride : Nat)
    (backward : Bool) (row group : Nat) : ProgramIO Numerics.Tagged Numerics.Tagged :=
  Interface.programIO (Numerics.operations (Numerics.softwarePrimitives cfg)) r
    qStride cosineStride sineStride backward row group

/-- Independent native FMA pair, with the original subtraction/addition
operand order, negation profile and rounding configuration. -/
def native (cfg : FP.Config) (r : Request) (backward : Bool)
    (values : Interface.Inputs Numerics.Tagged r) : Interface.Outputs Numerics.Tagged r :=
  Interface.result (fun h c =>
    let i := Interface.lane r c
    Numerics.leaf (Numerics.pair (Numerics.softwarePrimitives cfg) backward (decide (r.half ≤ c.val))
      (Interface.query values h (Interface.left r i)).word (Interface.query values h (Interface.right r i)).word
      (Interface.cosine values i).word (Interface.sine values i).word))

theorem fast_rope_fp32_correctness (cfg : FP.Config) (r : Request)
    (qStride cosineStride sineStride : Nat) (backward : Bool) (row group : Nat)
    (launch : r.Launched row group) (strideEq : row * qStride = row * (r.heads * r.dimension))
    (cosineEq : cosineStride = r.half) (sineEq : sineStride = r.half) :
    nativeIO cfg r qStride cosineStride sineStride backward row group ⊨ native cfg r backward :=
  Interface.native_implements cfg r qStride cosineStride sineStride backward row group launch strideEq cosineEq sineEq

#axiomsClean fast_rope_fp32_correctness
#specNonCircular native avoiding [Kernels.rope, Source.source, FP.Structural.exec]
#stmtSurfaceSubset fast_rope_fp32_correctness ⊆
  [nativeIO, native, ProgramIO.Implements, FP.Config, Numerics.Tagged, Numerics.instInhabitedTagged,
    Request, Request.Launched, Request.heads, Request.dimension, Request.half]
end
end VeriTile.Bench.Repairs.FastRopeEmbedding
