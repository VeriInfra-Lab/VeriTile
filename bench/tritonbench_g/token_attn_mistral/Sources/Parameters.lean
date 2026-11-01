import VeriTile.Triton

namespace VeriTile.Bench.Repairs.TokenAttnMistral
open VeriTile.Triton

structure Parameters where
  (prob value output : RegionName)
  (tokens request start length attentionLength : Region .nat)
  unusedStart : RegionName
  (requestStride sequenceStride attentionStartStride attentionLengthStride : Nat)
  (requestBatchStride requestTokenStride probHeadStride probTokenStride : Nat)
  (valueBatchStride valueHeadStride valueDimStride : Nat)
  (outputBatchStride outputHeadStride outputDimStride : Nat)
  (kvGroup window width block : Nat)


end VeriTile.Bench.Repairs.TokenAttnMistral
