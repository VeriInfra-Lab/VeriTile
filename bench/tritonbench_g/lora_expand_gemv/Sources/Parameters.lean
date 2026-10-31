import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton
structure Parameters where
  (input weight output : RegionName)
  indices : Region .int
  (inner outputSize splits inputBatchStride inputInnerStride loraStride weightRowStride
    weightInnerStride outputBatchStride outputStride blockRows blockInner : Nat)
  (evenK addInputs castType : Bool)

end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
