import VeriTile.Triton

namespace VeriTile.Bench.Repairs.TokenAttnReduceV
open VeriTile.Triton

structure Parameters where
  (prob value output : RegionName)
  (tokens request start length : Region .nat)
  (requestStride startStride lengthStride : Nat)
  (requestBatchStride requestTokenStride probHeadStride probTokenStride : Nat)
  (valueBatchStride valueHeadStride valueDimStride : Nat)
  (outputBatchStride outputHeadStride outputDimStride : Nat)
  (kvGroup width block : Nat)

end VeriTile.Bench.Repairs.TokenAttnReduceV
