import VeriTile.Triton
import VeriTile.Meta.StatementAudit
namespace VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
open VeriTile.Triton

structure Parameters where
  (x weight y : RegionName)
  (xSeqStride xHeadStride xDimStride ySeqStride yHeadStride yDimStride weightHeadStride weightDimStride columns block : Nat)
  epsilon : ℝ

end VeriTile.Bench.TritonBenchG.LayernormFwdTriton.FPStages
