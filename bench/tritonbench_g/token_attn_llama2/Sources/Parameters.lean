import VeriTile.Triton

namespace VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
open VeriTile.Triton

structure Parameters where
  (q k : RegionName)
  scale : ℝ
  (locations starts lengths : Region .nat)
  output : RegionName
  (maxLength locationBatchStride locationSequenceStride queryBatchStride
    queryHeadStride queryDimStride keyBatchStride keyHeadStride keyDimStride
    outputHeadStride outputSequenceStride groups dimensions tokens : Nat)

end VeriTile.Bench.Repairs.TokenAttnLlama2.FPValue
