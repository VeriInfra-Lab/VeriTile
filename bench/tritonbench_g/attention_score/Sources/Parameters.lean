import bench.tritonbench_g.attention_score.SourceDefinitions

namespace VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSource
open VeriTile.Triton

structure Parameters where
  (Q K M Out : RegionName)
  (queryBatch queryHead queryRow queryDim : Nat)
  (keyBatch keyHead keyRow keyDim : Nat)
  (outputBatch outputHead outputInner : Nat)
  (batches heads kvHeads queryContext roundedContext keyContext : Nat)
  (offset size : Int)
  scale : ℝ
  (window complement evenM evenN : Bool)

end VeriTile.Bench.Repairs.AttentionScoreTileChecked.Reused.RealSource
