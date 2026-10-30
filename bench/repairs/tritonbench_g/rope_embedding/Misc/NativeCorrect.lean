import bench.repairs.tritonbench_g.rope_embedding.Proofs.Interface
import bench.repairs.tritonbench_g.rope_embedding.Proofs.SourceBinding

/-! Auxiliary source-to-native correctness. This ordered native formula is
retained under its historical theorem name; the program pair is in FPEquiv. -/
namespace VeriTile.Bench.Repairs.RopeEmbedding
open VeriTile.Triton FP.Structural
noncomputable section

/-- Independent ordered native pair, including the exact multiply/FMA
operand order and backward sine negation. No numerical rewrite is assumed. -/
def native (cfg : FP.Config) (r : Request) (back : Bool)
    (values : FastRopeEmbedding.Interface.Inputs Numerics.Tagged r) :
    FastRopeEmbedding.Interface.Outputs Numerics.Tagged r :=
  FastRopeEmbedding.Interface.result (fun h c =>
    let i := FastRopeEmbedding.Interface.lane r c
    Numerics.leaf (Numerics.pair (Numerics.softwarePrimitives cfg) back (decide (r.half ≤ c.val))
      (FastRopeEmbedding.Interface.query values h (FastRopeEmbedding.Interface.left r i)).word
      (FastRopeEmbedding.Interface.query values h (FastRopeEmbedding.Interface.right r i)).word
      (FastRopeEmbedding.Interface.cosine values i).word (FastRopeEmbedding.Interface.sine values i).word))

theorem rope_embedding_kernel_fp32_equiv (cfg : FP.Config) (r : Request) (back : Bool)
    (row group : Nat) (launch : r.Launched row group) :
    (nativeIO cfg r back row group).Implements (native cfg r back) :=
  Interface.native_implements cfg r back row group launch

#specNonCircular native avoiding [Kernels.rope, Source.source, FP.Structural.exec]
#stmtSurfaceSubset rope_embedding_kernel_fp32_equiv ⊆
  [nativeIO, native, ProgramIO.Implements, FP.Config, Numerics.Tagged, Numerics.instInhabitedTagged,
    Request, Request.Launched]
#axiomsClean rope_embedding_kernel_fp32_equiv
end
end VeriTile.Bench.Repairs.RopeEmbedding
