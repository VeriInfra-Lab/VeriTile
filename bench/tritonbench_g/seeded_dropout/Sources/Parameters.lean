import VeriTile.Triton.DSL
namespace VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource
open VeriTile.Triton
inductive Storage where
  | fp32 | fp16 | bf16 | fp64 | f8e4 | f8e5
  deriving DecidableEq

def SeedCarrier : Random.IntegerType → Type
  | .i32 | .i64 => Int
  | .u32 | .u64 => Nat

def seedInt (kind : Random.IntegerType) (seed : SeedCarrier kind) : Int :=
  match kind, seed with
  | .i32, s | .i64, s => s
  | .u32, s | .u64, s => Int.ofNat s

end VeriTile.Bench.TritonBenchG.SeededDropout.TypedSource
