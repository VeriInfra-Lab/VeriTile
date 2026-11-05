import VeriTile.Triton

namespace VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
open VeriTile.Triton

structure Parameters where
  (logics values output : RegionName)
  locations : Region .int
  (starts lengths : Region .nat)
  (maxLength logicHeadStride logicSequenceStride valueBatchStride valueHeadStride valueDimStride
    outputBatchStride outputHeadStride outputDimStride locationBatchStride locationSequenceStride
    dimensions tokens : Nat)
  sentinel : Int

end VeriTile.Bench.Repairs.SoftmaxReduceVNegInfChecked.FPValue
