import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
open VeriTile.Triton
inductive Storage where
  | fp16 | bf16 | fp32 | fp64 | f8e4 | f8e5 | int64
  deriving DecidableEq, Repr

def Storage.tag : Storage → TileDType
  | .fp16 => .fp16 | .bf16 => .bf16 | .fp32 | .fp64 => .real
  | .f8e4 => .f8e4 | .f8e5 => .f8e5 | .int64 => .int

structure Parameters where
  (input output : RegionName)
  selection : Region .bool
  prefixSum : Region .nat
  (size width : Nat)

end VeriTile.Bench.TritonBenchG.MaskedSelect.TypedCopy
