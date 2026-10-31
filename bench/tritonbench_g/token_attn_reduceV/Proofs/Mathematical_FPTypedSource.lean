import VeriTile.Triton
namespace VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
open VeriTile.Triton

structure Parameters where
  (prob value output : RegionName)
  (tokens request start length : Region .nat)
  (requestBatchStride requestTokenStride probHeadStride probTokenStride : Nat)
  (valueBatchStride valueHeadStride valueDimStride : Nat)
  (outputBatchStride outputHeadStride outputDimStride : Nat)
  (kvGroup width block : Nat)

end VeriTile.Bench.TritonBenchG.TokenAttnReduceV.FPStages
