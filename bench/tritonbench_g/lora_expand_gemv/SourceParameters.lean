import VeriTile.Triton.Core.Types

/-! Source-only call parameters. Declaration names, fields and domains are unchanged. -/
namespace VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
open VeriTile.Triton

inductive InputProfile where
  | halfHalf | halfBfloat | bfloatHalf | bfloatBfloat | singleHalf | singleBfloat
  deriving DecidableEq, Repr

structure Parameters where
  (input weight output : RegionName)
  indices : Region .int
  (inner outputSize splits inputBatchStride inputInnerStride loraStride weightRowStride
    weightInnerStride outputBatchStride outputStride blockRows blockInner : Nat)
  (evenK addInputs castType : Bool)

end VeriTile.Bench.TritonBenchG.LoraExpandGemv.Branches
